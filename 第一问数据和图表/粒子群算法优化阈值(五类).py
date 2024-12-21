import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from pyswarm import pso
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据集
data = pd.read_excel('第三段时间序列分析数据.xlsx')

# 将数据集中相关列提取为数组
speed_data = data['平均速度 (千米/小时)'].values
density_data = data['车辆密度 (辆/公里)'].values
true_labels = data['拥堵等级'].values

# 定义类别映射，增加五个类别
label_mapping = {'基本畅通': 0, '缓行': 1, '轻度拥堵': 2, '拥堵': 3, '严重拥堵': 4}

# 将标签映射为数值
true_labels = np.array([label_mapping[label] for label in true_labels])

# 定义分类函数，基于五个类别的阈值
def classify(speed, density, thresholds):
    speed_th1, speed_th2, speed_th3, speed_th4, density_th1, density_th2, density_th3, density_th4 = thresholds
    if speed > speed_th1 and density < density_th1:
        return 0  # 基本畅通
    elif speed_th2 < speed <= speed_th1 and density_th2 < density <= density_th1:
        return 1  # 缓行
    elif speed_th3 < speed <= speed_th2 and density_th3 < density <= density_th2:
        return 2  # 轻度拥堵
    elif speed_th4 < speed <= speed_th3 and density_th4 < density <= density_th3:
        return 3  # 拥堵
    else:
        return 4  # 严重拥堵

# 定义适应度函数
def fitness(thresholds):
    predictions = []
    for speed, density in zip(speed_data, density_data):
        predictions.append(classify(speed, density, thresholds))
    accuracy = accuracy_score(true_labels, predictions)
    return -accuracy  # 负的准确率，因为pso是最小化问题

# PSO 每次迭代时保存当前的准确率
accuracy_history = []

def fitness_with_history(thresholds):
    global accuracy_history
    accuracy = -fitness(thresholds)
    accuracy_history.append(accuracy)
    return -accuracy

# 设置阈值的上下界 (包括四个速度和四个密度阈值)
lb = [20, 30, 40, 50, 60, 70, 80, 90]  # 下界：四个速度和四个密度的下界
ub = [100, 90, 80, 70, 140, 130, 120, 110]  # 上界：四个速度和四个密度的上界

# 使用粒子群优化算法寻找最佳阈值, 并返回所有迭代的准确率
best_thresholds, best_accuracy = pso(fitness_with_history, lb, ub, swarmsize=30, maxiter=50)

# 输出最优结果
print(f"最优阈值: {best_thresholds}")
print(f"最优准确率: {-best_accuracy}")

# 使用最优阈值进行预测
predictions = []
for speed, density in zip(speed_data, density_data):
    predictions.append(classify(speed, density, best_thresholds))

# 添加预测结果到数据集
data['预测拥堵等级'] = predictions
data['预测拥堵等级'] = data['预测拥堵等级'].map({0: '基本畅通', 1: '缓行', 2: '轻度拥堵', 3: '拥堵', 4: '严重拥堵'})

# 保存结果到Excel
data.to_excel('pso_predicted_results.xlsx', index=False)

# 可视化真实标签 vs 预测标签
plt.figure(figsize=(10, 6))
plt.plot(data.index, true_labels, label='真实标签', linestyle='-', marker='o')
plt.plot(data.index, predictions, label='预测标签', linestyle='--', marker='x')
plt.xlabel('样本索引')
plt.ylabel('拥堵等级')  # 使用数值来表示不同拥堵等级
plt.title('真实标签 vs 预测标签')
plt.yticks([0, 1, 2, 3, 4], ['基本畅通', '缓行', '轻度拥堵', '拥堵', '严重拥堵'])  # 设置Y轴标签
plt.legend()
plt.savefig('pso_predicted_vs_true.png')
plt.show()

# 绘制准确率随迭代次数的变化曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, marker='o')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.title('粒子群优化迭代准确率变化')
plt.grid(True)
plt.savefig('pso_accuracy_iterations.png')
plt.show()
