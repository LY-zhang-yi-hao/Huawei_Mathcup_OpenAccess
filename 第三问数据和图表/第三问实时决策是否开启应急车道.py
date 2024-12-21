#1.根据阈值准则判断是否需要开启应急车道
import pandas as pd

# 创建数据框

df=pd.read_excel('总和数据.xlsx')
# 定义拥堵状态编号
congestion_levels = {
    '基本畅通': 1,
    '缓行': 2,
    '轻度拥堵': 3,
    '拥堵': 4,
    '严重拥堵': 5
}

# 将拥堵等级映射到编号
df['拥堵等级编号'] = df['拥堵等级'].map(congestion_levels)

# 定义阈值条件
flow_threshold = 5000
density_threshold = 140
speed_threshold = 30
level_threshold = 4


# 定义是否开启拥堵状态的函数
def check_congestion(row):
    conditions = [
        row['交通流量（辆/小时）'] > flow_threshold,  # 流量大于5000
        row['车辆密度 (辆/公里)'] > density_threshold,  # 密度大于140
        row['平均速度 (千米/小时)'] < speed_threshold,  # 速度小于30
        row['拥堵等级编号'] >= level_threshold  # 拥堵状态编号大于等于4
    ]

    # 至少满足三个条件
    if sum(conditions) >= 3:
        return True
    else:
        return False


# 应用函数，判断是否开启拥堵状态
df['开启拥堵状态'] = df.apply(check_congestion, axis=1)
df.to_excel("是否开启应急车道结果.xlsx")
# 显示结果
print(df[['平均速度 (千米/小时)', '车辆密度 (辆/公里)', '交通流量（辆/小时）', '拥堵等级', '开启拥堵状态']])