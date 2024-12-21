import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 设置中文字体以避免字体缺失警告
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = pd.read_excel("第三段数据.xlsx")

# 选择相关列
columns_to_use = ['平均速度 (千米/小时)', '车辆密度 (辆/公里)', '交通流量（辆/小时）']
data = data[columns_to_use]

# 数据平滑处理（可选），减少噪声影响
# data = data.rolling(window=3).mean().dropna()

# 数据归一化处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 准备训练集和测试集
n_past = 30  # 使用过去30个数据点
n_future = 15  # 预测未来15个数据点（即未来30分钟，每隔2分钟一个点）
n_features = scaled_data.shape[1]  # 特征数

X_train = []
y_train = []

for i in range(n_past, len(scaled_data) - n_future + 1):
    X_train.append(scaled_data[i - n_past:i, :])
    y_train.append(scaled_data[i:i + n_future, :])

X_train, y_train = np.array(X_train), np.array(y_train)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(n_past, n_features)))  # 调整LSTM单元数量
model.add(Dropout(0.2))  # 添加Dropout防止过拟合
model.add(LSTM(64, return_sequences=False))  # 第二个LSTM层
model.add(Dropout(0.2))
model.add(Dense(n_future * n_features))  # 输出层，预测未来多个时间步
model.compile(optimizer='adam', loss='mean_squared_error')

# 模型训练
history = model.fit(X_train, y_train.reshape(y_train.shape[0], n_future * n_features), epochs=50, batch_size=32,
                    validation_split=0.2, verbose=1)

# 预测未来30分钟
X_test = scaled_data[-n_past:, :].reshape(1, n_past, n_features)  # 使用最后n_past个数据点作为测试
predictions_lstm = model.predict(X_test)

# 将预测结果恢复到原始缩放比例
predictions_lstm = scaler.inverse_transform(predictions_lstm.reshape(n_future, n_features))

# 将预测结果转换为 DataFrame 并设置索引
predictions_lstm = pd.DataFrame(predictions_lstm, columns=columns_to_use)
predictions_lstm.index = range(276, 306, 2)  # 预测的时间索引，276分钟到306分钟，每隔2分钟

# 打印预测结果
print(predictions_lstm)

# 保存预测结果为Excel文件
predictions_lstm.to_excel('lstm_predictions_optimized.xlsx')

# 可视化预测结果
for column in columns_to_use:
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(data)), data[column], label="历史数据", color='blue')
    plt.plot(predictions_lstm.index, predictions_lstm[column], label="预测数据", linestyle='dashed', color='orange')
    plt.xlabel("时间 (分钟)")
    plt.ylabel(column)
    plt.title(f"LSTM 预测 - {column}")
    plt.legend()

    # 处理文件名中的特殊字符（去除括号和斜杠）
    safe_column_name = column.replace("(", "").replace(")", "").replace("/", "")

    plt.savefig(f'{safe_column_name}_lstm_prediction_optimized.png')  # 保存图片
    plt.show()

