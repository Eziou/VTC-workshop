import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
import time  # 导入 time 库

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/home/zoufangming/Documents/Guangdian/实验代码/完整版/预训练/data/data.csv", names=col_names)
# Randomly sample 1000 samples
# data = data.sample(n=2000, random_state=40).reset_index(drop=True)
# print(f"Total samples after random sampling: {len(data)}")

# 提取特征和标签
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
dataset_Y = data["label"].values.flatten()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)


# train_data = pd.read_csv("/home/zoufangming/Documents/Guangdian/实验代码/小样本数据集/ratio_1_1_train_1000.csv", names=col_names, 
#                         header=0)
# test_data = pd.read_csv("/home/zoufangming/Documents/Guangdian/实验代码/小样本数据集/ratio_1_1_test_200.csv", names=col_names, 
#                         header=0)

# x_train = train_data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
# y_train = train_data["label"].values.flatten()
# x_test = test_data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
# y_test = test_data["label"].values.flatten()
# 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义要搜索的C参数范围
C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 20, 30]

# LinearSVC网格搜索
print("\n执行LinearSVC网格搜索...")
linear_param_grid = {'C': C_values}
linear_grid = GridSearchCV(
    LinearSVC(penalty='l2', random_state=42, dual=False), 
    param_grid=linear_param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

start_grid_time = time.time()
linear_grid.fit(x_train, y_train)
end_grid_time = time.time()
grid_time = end_grid_time - start_grid_time

print(f"\nLinearSVC 网格搜索完成，耗时: {grid_time:.4f} 秒")
print(f"最佳参数: {linear_grid.best_params_}")
print(f"最佳准确率: {linear_grid.best_score_:.4f}")

# SVC网格搜索
print("\n执行SVC网格搜索...")
svc_param_grid = {'C': C_values}
svc_grid = GridSearchCV(
    SVC(kernel='linear', random_state=42), 
    param_grid=svc_param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

start_grid_time = time.time()
svc_grid.fit(x_train, y_train)
end_grid_time = time.time()
grid_time = end_grid_time - start_grid_time

print(f"\nSVC 网格搜索完成，耗时: {grid_time:.4f} 秒")
print(f"最佳参数: {svc_grid.best_params_}")
print(f"最佳准确率: {svc_grid.best_score_:.4f}")

# 使用最佳参数的LinearSVC模型
best_linear_svc = LinearSVC(C=linear_grid.best_params_['C'], penalty='l2', random_state=42, dual=False)
start_train_time = time.time()
print('\nStart training LinearSVC with best parameters...')
best_linear_svc.fit(x_train, y_train)
end_train_time = time.time()
linear_train_time = end_train_time - start_train_time

# 为LinearSVC计算总推理时间和每个样本的平均推理时间
start_predict_time = time.time()
print('Start predicting with LinearSVC...')
linear_y_pred = best_linear_svc.predict(x_test)
end_predict_time = time.time()
linear_predict_time = end_predict_time - start_predict_time
linear_per_sample_time = linear_predict_time / len(x_test)  # 每个样本的平均推理时间

# 使用最佳参数的SVC模型
best_svc = SVC(C=svc_grid.best_params_['C'], kernel='linear', random_state=42)
start_train_time = time.time()
print('\nStart training SVC with best parameters...')
best_svc.fit(x_train, y_train)
end_train_time = time.time()
svc_train_time = end_train_time - start_train_time

# 为SVC计算总推理时间和每个样本的平均推理时间
start_predict_time = time.time()
print('Start predicting with SVC...')
svc_y_pred = best_svc.predict(x_test)
end_predict_time = time.time()
svc_predict_time = end_predict_time - start_predict_time
svc_per_sample_time = svc_predict_time / len(x_test)  # 每个样本的平均推理时间

# 计算和打印LinearSVC的评估指标
linear_accuracy = accuracy_score(y_test, linear_y_pred)
linear_recall = recall_score(y_test, linear_y_pred)
linear_precision = precision_score(y_test, linear_y_pred)
linear_f1 = f1_score(y_test, linear_y_pred)

print('\n--- LinearSVC with Best Parameters ---')
print(f'Best C: {linear_grid.best_params_["C"]}')
print(f'Accuracy: {linear_accuracy:.4f}')
print(f'Recall: {linear_recall:.4f}')
print(f'Precision: {linear_precision:.4f}')
print(f'F1 Score: {linear_f1:.4f}')
print(f"Training time: {linear_train_time:.4f} seconds")
print(f"Total prediction time: {linear_predict_time:.4f} seconds")
print(f"Average prediction time per sample: {linear_per_sample_time*1000:.4f} ms")

# 计算和打印SVC的评估指标
svc_accuracy = accuracy_score(y_test, svc_y_pred)
svc_recall = recall_score(y_test, svc_y_pred)
svc_precision = precision_score(y_test, svc_y_pred)
svc_f1 = f1_score(y_test, svc_y_pred)

print('\n--- SVC with Best Parameters ---')
print(f'Best C: {svc_grid.best_params_["C"]}')
print(f'Accuracy: {svc_accuracy:.4f}')
print(f'Recall: {svc_recall:.4f}')
print(f'Precision: {svc_precision:.4f}')
print(f'F1 Score: {svc_f1:.4f}')
print(f"Training time: {svc_train_time:.4f} seconds")
print(f"Total prediction time: {svc_predict_time:.4f} seconds")
print(f"Average prediction time per sample: {svc_per_sample_time*1000:.4f} ms")
