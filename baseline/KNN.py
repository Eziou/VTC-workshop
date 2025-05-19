import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time  # 导入 time 库

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/Users/han/Charging station/data/data.csv", names=col_names)

# 提取特征和标签
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
dataset_Y = data["label"].values.flatten()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义 KNN 模型
knn_model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # 添加n_neighbors参数
    'weights': ['uniform', 'distance'],  # 添加weights参数
    'metric': ['euclidean', 'manhattan']  # 添加metric参数
}

# 使用 GridSearchCV 进行超参数优化
grid_search = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)
start_train_time = time.time()  # 记录训练开始时间
print('Start training with grid search...')

grid_search.fit(x_train, y_train)

end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时间

# 输出最优参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最优参数进行预测
best_model = grid_search.best_estimator_

# 记录预测时间
start_predict_time = time.time()  # 记录开始预测的时间
print('Start predicting...')
y_pred = best_model.predict(x_test)
end_predict_time = time.time()  # 记录预测结束时间
predict_time = end_predict_time - start_predict_time  # 计算预测时间

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估指标
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')

# 打印训练和预测时间
print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")