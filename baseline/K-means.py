import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time  # 导入 time 库
from scipy.stats import mode

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/Users/han/Charging station/data/data.csv", names=col_names)

# 检查并清理数据中的 None 值
data.dropna(inplace=True)

# 提取特征
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset_X)

# 定义 KMeans 模型
kmeans = KMeans(n_clusters=2, random_state=42)

# 使用 GridSearchCV 进行超参数优化
param_grid = {'n_clusters': [2, 3, 4, 5]}
grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1, error_score='raise')

# 记录训练开始时间
start_train_time = time.time()  # 记录训练开始时间
print('Start training with grid search...')

grid_search.fit(X_scaled)

# 记录训练结束时间
end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时间

# 输出最优参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最优参数进行预测
best_model = grid_search.best_estimator_

# 记录预测开始时间
start_predict_time = time.time()  # 记录开始预测的时间
print('Start predicting...')

# 预测聚类标签
y_pred = best_model.predict(X_scaled)

# 记录预测结束时间
end_predict_time = time.time()  # 记录预测结束的时间
predict_time = end_predict_time - start_predict_time  # 计算预测时间

# 将聚类结果映射到真实标签
labels_true = data["label"].values
labels_pred = np.zeros_like(labels_true)
for i in range(best_model.n_clusters):
    mask = (y_pred == i)
    labels_pred[mask] = mode(labels_true[mask])[0][0]  # 确保返回的是模式值而不是元组

# 计算评估指标
accuracy = accuracy_score(labels_true, labels_pred)
recall = recall_score(labels_true, labels_pred, average='weighted')
precision = precision_score(labels_true, labels_pred, average='weighted')
f1 = f1_score(labels_true, labels_pred, average='weighted')

# 打印评估指标
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')

# 打印训练和预测时间
print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")
