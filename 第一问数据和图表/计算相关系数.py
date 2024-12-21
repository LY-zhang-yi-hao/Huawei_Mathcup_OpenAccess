import pandas as pd
from scipy.stats import pearsonr
import numpy as np

# 读取数据
df = pd.read_excel("第三段数据.xlsx")

# 创建拥堵等级的编码字典
congestion_mapping = {
    '畅通': 1,
    '缓行': 2,
    '轻度拥堵': 3,
    '中度拥堵': 4,
    '严重拥堵': 5
}

# 对因变量 '拥堵等级' 进行编码
df['拥堵等级编码'] = df['拥堵等级'].map(congestion_mapping)

# 选择自变量列和因变量编码列
variables = ['平均速度 (千米/小时)', '交通流量（辆/小时）', '车辆密度 (辆/公里)', '拥堵等级编码']

# 计算相关性矩阵
correlation_matrix = df[variables].corr()

# 初始化p值矩阵（用与相关性矩阵相同的结构）
p_value_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)

# 逐一计算每对变量之间的p值
for var1 in variables:
    for var2 in variables:
        if var1 != var2:  # 自己与自己比较的相关性已经在 correlation_matrix 中，所以跳过
            _, p_value = pearsonr(df[var1], df[var2])
            p_value_matrix.loc[var1, var2] = p_value
        else:
            p_value_matrix.loc[var1, var2] = np.nan  # 同一变量的p值设为NaN

# 打印相关性矩阵和p值矩阵
print("相关性矩阵:")
print(correlation_matrix)

print("\nP值矩阵:")
print(p_value_matrix)
