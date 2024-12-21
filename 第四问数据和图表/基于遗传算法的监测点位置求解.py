import numpy as np
import random
import matplotlib.pyplot as plt

# 参数设置
num_locations = 15  # 总共15个观测点
max_monitor_points = 15  # 允许的最大监控点数
min_monitor_points = 8  # 允许的最小监控点数
max_cost = 300000  # 最大预算
min_distance = 2  # 最小监控点间距（以100米为单位）

# 监控点参数数据（速度、密度、流量）
speed = np.array(
    [58.44, 54.03, 54.13, 57.56, 52.53, 54.47, 53.46, 59.38, 56.39, 60.04, 57.24, 53.24, 55.94, 76.27, 66.20])
density = np.array([82, 85, 85, 83, 91, 86, 87, 90, 82, 88, 83, 86, 85, 72, 75])
flow = np.array(
    [4791.93, 4592.53, 4600.68, 4777.51, 4780.45, 4684.51, 4650.82, 5344.48, 4624.35, 5283.10, 4750.69, 4578.81,
     4754.56, 5491.30, 4965.05])

# 成本数据（假设随机生成的成本）
# cost = np.random.randint(50, 150, size=num_locations)
cost = np.random.randint(10000, 30000, size=num_locations)

# 归一化函数，避免除以 0 的情况
def normalize(x):
    min_x = np.min(x)
    max_x = np.max(x)
    if max_x == min_x:
        return np.zeros_like(x)  # 如果所有值相同，返回全 0
    return (x - min_x) / (max_x - min_x)

# 适应度函数（计算效果评分和成本）
def fitness(individual):
    selected_indices = np.where(individual == 1)[0]
    if len(selected_indices) == 0:
        return -100  # 如果没有选中任何监控点，适应度为一个较低值
    # 计算效果评分
    f_speed = normalize(speed[selected_indices])
    f_density = normalize(density[selected_indices])
    f_flow = normalize(flow[selected_indices])
    E = 0.3 * f_speed + 0.3 * f_density + 0.4 * f_flow  # 权重系数分别为0.3, 0.3, 0.4
    total_effect = np.sum(E)

    # 计算总成本
    total_cost = np.sum(cost[selected_indices])

    # 引入适应度基线，避免适应度为0
    w1, w2 = 1, 0.1
    penalty = 0

    # 预算约束惩罚
    if total_cost > max_cost:
        penalty += (total_cost - max_cost) * 0.05  # 惩罚项（允许轻微超预算）

    # 检查监控点间距
    if not valid(individual):
        penalty += 50  # 违反监控点间距的惩罚

    return w1 * total_effect - w2 * total_cost - penalty  # 加上惩罚项

# 生成初始种群时确保监控点数量在允许范围内
def generate_population(pop_size):
    population = []
    for _ in range(pop_size):
        individual = np.zeros(num_locations)
        num_points = random.randint(min_monitor_points, max_monitor_points)
        selected_indices = random.sample(range(num_locations), num_points)
        individual[selected_indices] = 1
        population.append(individual)
    return np.array(population)

# 选择操作（轮盘赌）
def selection(population, fitness_values):
    fitness_values = np.nan_to_num(fitness_values, nan=0.1, posinf=0.1, neginf=0.1)
    if np.all(fitness_values <= 0):
        fitness_values = np.ones_like(fitness_values)  # 避免总权重为0的情况
    selected = random.choices(population, weights=fitness_values, k=len(population))
    return np.array(selected)

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, num_locations)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return repair(child1), repair(child2)

# 变异操作
def mutation(individual, mutation_rate=0.01):
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]  # 翻转0和1
    return repair(individual)

# 修正个体，确保监控点数量在允许范围内
def repair(individual):
    num_selected = np.sum(individual)
    if num_selected < min_monitor_points:
        additional_indices = random.sample(list(np.where(individual == 0)[0]), min_monitor_points - int(num_selected))
        individual[additional_indices] = 1
    elif num_selected > max_monitor_points:
        remove_indices = random.sample(list(np.where(individual == 1)[0]), int(num_selected) - max_monitor_points)
        individual[remove_indices] = 0
    return individual

# 约束条件：保证监控点数量和间距
def valid(individual):
    num_selected = np.sum(individual)
    if num_selected < min_monitor_points or num_selected > max_monitor_points:
        return False
    selected_indices = np.where(individual == 1)[0]
    for i in range(len(selected_indices) - 1):
        if selected_indices[i + 1] - selected_indices[i] < min_distance:
            return False
    return True

# 遗传算法主程序
def genetic_algorithm(pop_size=50, generations=100):
    population = generate_population(pop_size)
    best_individual = None
    best_fitness = -np.inf
    fitness_history = []

    for gen in range(generations):
        # 计算适应度值
        fitness_values = np.array([fitness(ind) for ind in population])
        best_gen_fitness = np.max(fitness_values)
        best_gen_individual = population[np.argmax(fitness_values)]

        # 更新最佳个体
        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_individual = best_gen_individual

        # 选择、交叉和变异
        selected_population = selection(population, fitness_values)
        new_population = []
        for i in range(0, len(selected_population), 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))
        population = np.array(new_population)

        fitness_history.append(best_fitness)
        print(f"Generation {gen + 1}: Best fitness = {best_fitness}")

    # 绘制迭代曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of Fitness Over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_evolution.png')
    plt.show()

    return best_individual, best_fitness, fitness_history

# 运行遗传算法
best_solution, best_solution_fitness, fitness_history = genetic_algorithm()
selected_monitor_points = np.where(best_solution == 1)[0]

# 输出最优解
print("最优监控点布局:", selected_monitor_points)
print("最优适应度值:", best_solution_fitness)

# 导出适应度结果
np.savetxt('fitness_history.csv', fitness_history, delimiter=',', header='Fitness')
