import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import LinearSVC
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

# 定义 LinearSVC 模型
svm_model = LinearSVC(C=1000, penalty='l1', random_state=42, dual=False, verbose=0)

# 记录训练时间
start_train_time = time.time()  # 记录开始训练的时间
print('Start training...')
svm_model.fit(x_train, y_train)
end_train_time = time.time()  # 记录训练结束的时间
train_time = end_train_time - start_train_time  # 计算训练时间

# 记录预测时间
start_predict_time = time.time()  # 记录开始预测的时间
print('Start predicting...')
y_pred = svm_model.predict(x_test)
end_predict_time = time.time()  # 记录预测结束的时间
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
