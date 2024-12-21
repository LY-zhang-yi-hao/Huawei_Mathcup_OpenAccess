import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei']  # SimHei 是支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
df = pd.read_excel('第三段时间序列分析数据.xlsx')

# 选择需要进行预测的时间序列数据
columns_to_predict = ['平均速度 (千米/小时)', '车辆密度 (辆/公里)', '交通流量（辆/小时）']

# 创建一个存储预测结果的DataFrame
predictions = pd.DataFrame()


# 进行ARMA模型的p, q参数网格搜索
def grid_search_arma(series, max_p, max_q):
    best_aic = np.inf
    best_order = None
    best_model = None
    aic_values = np.zeros((max_p + 1, max_q + 1))

    # 网格搜索p, q
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, 0, q))
                model_fit = model.fit()
                aic = model_fit.aic
                aic_values[p, q] = aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, q)
                    best_model = model_fit
            except:
                aic_values[p, q] = np.nan  # 如果模型报错，则设为NaN，忽略该组合
                continue

    return best_order, best_model, aic_values


# 保存并可视化热力图
def plot_aic_heatmap(aic_values, column_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(aic_values, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=range(aic_values.shape[1]),
                yticklabels=range(aic_values.shape[0]))
    plt.title(f'{column_name} 的ARMA模型AIC值热力图')
    plt.xlabel('q值')
    plt.ylabel('p值')

    # 处理文件名，去除特殊字符
    clean_column_name = column_name.replace(' ', '_').replace('（', '_').replace('）', '').replace('/', '_').replace('\\',
                                                                                                                   '_')

    plt.savefig(f'{clean_column_name}_aic_heatmap.png')
    plt.show()


# 对每个列进行ARMA预测和网格搜索
for column in columns_to_predict:
    series = df[column]

    # 进行网格搜索，最大p和q值设为5
    best_order, best_model, aic_values = grid_search_arma(series, max_p=5, max_q=5)

    # 输出最佳模型的p, q值
    print(f'{column} 最佳ARMA模型 p={best_order[0]}, q={best_order[1]}，AIC值为 {best_model.aic}')

    # 预测未来30分钟（15个时间点，间隔2分钟）
    forecast = best_model.forecast(steps=15)

    # 存储预测结果
    predictions[column + '_预测'] = forecast

    # 绘制AIC值热力图
    plot_aic_heatmap(aic_values, column)

# 保存预测结果到Excel
predictions.index = range(276, 306, 2)  # 预测的索引，即未来 30 分钟内，每隔 2 分钟一个数据点
predictions.to_excel('arma_predictions_30min.xlsx', index=True)

# 可视化并保存预测结果
for column in columns_to_predict:
    plt.figure(figsize=(10, 5))
    plt.plot(df['时间 (分钟)'], df[column], label='历史数据')
    plt.plot(predictions.index, predictions[column + '_预测'], label='预测数据', linestyle='--')
    plt.title(f'{column} 的ARMA预测')
    plt.xlabel('时间 (分钟)')
    plt.ylabel(column)
    plt.legend()

    # 处理文件名，去除特殊字符
    clean_column_name = column.replace(' ', '_').replace('（', '_').replace('）', '').replace('/', '_').replace('\\', '_')
    plt.savefig(f'{clean_column_name}_arma_prediction_30min.png')
    plt.show()
