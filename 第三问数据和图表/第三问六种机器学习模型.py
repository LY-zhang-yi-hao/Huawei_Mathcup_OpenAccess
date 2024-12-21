
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
file_path = '是否开启应急车道结果.xlsx'  # 请替换为你的文件路径
data = pd.read_excel(file_path)

# 数据预处理：将开启拥堵状态编码为0和1
data['开启拥堵状态'] = data['开启拥堵状态'].astype(int)

# 选择自变量和因变量
X = data[['平均速度 (千米/小时)', '车辆密度 (辆/公里)', '交通流量（辆/小时）', '拥堵等级编号']]
y = data['开启拥堵状态']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化模型
models = {
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'BP Neural Network': MLPClassifier(random_state=42),
}

# 创建Stacking模型
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
models['Stacking'] = stacking_model

# 进行5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 保存交叉验证结果
cv_results = []
confusion_matrices_cv = []
roc_curves_cv = {}

for name, model in models.items():
    # Cross-validation predictions
    y_pred_cv = cross_val_predict(model, X, y, cv=cv, method='predict')
    y_prob_cv = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    # 计算评价指标
    accuracy_cv = accuracy_score(y, y_pred_cv)
    recall_cv = recall_score(y, y_pred_cv)
    f1_cv = f1_score(y, y_pred_cv)

    # 保存评价指标
    cv_results.append({
        'Model': name,
        'Accuracy': accuracy_cv,
        'Recall': recall_cv,
        'F1 Score': f1_cv
    })

    # 混淆矩阵
    confusion_matrices_cv.append(confusion_matrix(y, y_pred_cv))

    # ROC曲线
    fpr_cv, tpr_cv, _ = roc_curve(y, y_prob_cv)
    roc_auc_cv = auc(fpr_cv, tpr_cv)
    roc_curves_cv[name] = (fpr_cv, tpr_cv, roc_auc_cv)

# 展示交叉验证的结果
cv_results_df = pd.DataFrame(cv_results)
print(cv_results_df)

# 画出混淆矩阵
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.ravel()

for i, (name, matrix_cv) in enumerate(zip(models.keys(), confusion_matrices_cv)):
    sns.heatmap(matrix_cv, annot=True, fmt="d", ax=axes[i], cmap="Blues")
    axes[i].set_title(f"{name} CV Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# 画出ROC曲线
plt.figure(figsize=(10, 8))
for name, (fpr_cv, tpr_cv, roc_auc_cv) in roc_curves_cv.items():
    plt.plot(fpr_cv, tpr_cv, label=f'{name} (AUC = {roc_auc_cv:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Cross-Validation ROC Curve')
plt.legend(loc='lower right')
plt.show()
