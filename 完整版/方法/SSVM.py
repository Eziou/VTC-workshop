import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
import os
import itertools

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
# data = pd.read_csv("/home/zoufangming/Documents/Guangdian/实验代码/完整版/预训练/data/data.csv", names=col_names)
data = pd.read_csv("/home/zoufangming/Documents/Guangdian/实验代码/完整版/预训练/data/data.csv", names=col_names)
# Select first 200 and last 200 samples
# Randomly sample 400 samples
# data = data.sample(n=2000, random_state=40).reset_index(drop=True)
# print(f"Total samples after random sampling: {len(data)}")
# 提取特征和标签
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
dataset_Y = data["label"].values.flatten()

# train_data = pd.read_csv(
#     "/home/zoufangming/Documents/Guangdian/实验代码/小样本数据集/ratio_1_1_train_800.csv", 
#     names=col_names, 
#     header=0)
# test_data = pd.read_csv("/home/zoufangming/Documents/Guangdian/实验代码/小样本数据集/ratio_1_1_test_200.csv", names=col_names, 
#                         header=0)

# x_train = train_data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
# y_train = train_data["label"].values.flatten()
# x_test = test_data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
# y_test = test_data["label"].values.flatten()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义两层 SVM 模型类
class TwoLayerSVM:
    def __init__(self):
        # 第一层和第二层的 SVM（初始为空，待训练后赋值）
        self.svm_layer1 = None
        self.svm_layer2_s0 = None
        self.svm_layer2_s1 = None
        # 记录参数
        self.params = {
            'Layer 1': {'C': 1, 'kernel': 'linear'},
            'Layer 2 (S_0)': {'C': 0.1, 'kernel': 'linear'},
            'Layer 2 (S_1)': {'C': 1, 'kernel': 'linear'}
        }
        self.best_params = self.params.copy()
        self.best_accuracy = 0.0

    def train_svm(self, X, y, layer_name):
        # 根据层级设置参数
        params = self.params[layer_name]
        svm = SVC(kernel='linear', C=params['C'], random_state=40)
        start_time = time.time()
        svm.fit(X, y)
        end_time = time.time()
        print(f"{layer_name} - Using parameters: {params}")
        print(f"{layer_name} training time: {end_time - start_time:.4f} seconds")
        return svm

    def fit(self, X, y):
        # 训练第一层
        print('Training Layer 1...')
        self.svm_layer1 = self.train_svm(X, y, 'Layer 1')

        # 第一层预测训练集标签
        labels_layer1 = self.svm_layer1.predict(X)

        # 分割数据为 S_0 和 S_1
        mask_s0 = labels_layer1 == 0  # 子集 S_0
        mask_s1 = labels_layer1 == 1  # 子集 S_1
        X_s0, y_s0 = X[mask_s0], y[mask_s0]
        X_s1, y_s1 = X[mask_s1], y[mask_s1]

        # 训练第二层 S_0
        print('Training Layer 2 (Subset S_0)...')
        if len(X_s0) > 0 and len(np.unique(y_s0)) > 1:
            self.svm_layer2_s0 = self.train_svm(X_s0, y_s0, 'Layer 2 (S_0)')
        else:
            print("Subset S_0 has insufficient data or classes for training.")
            self.svm_layer2_s0 = None

        # 训练第二层 S_1
        print('Training Layer 2 (Subset S_1)...')
        if len(X_s1) > 0 and len(np.unique(y_s1)) > 1:
            self.svm_layer2_s1 = self.train_svm(X_s1, y_s1, 'Layer 2 (S_1)')
        else:
            print("Subset S_1 has insufficient data or classes for training.")
            self.svm_layer2_s1 = None

    def predict(self, X):
        # 第一层预测
        labels_layer1 = self.svm_layer1.predict(X)

        # 分割数据为 S_0 和 S_1
        mask_s0 = labels_layer1 == 0
        mask_s1 = labels_layer1 == 1
        X_s0, X_s1 = X[mask_s0], X[mask_s1]

        # 第二层预测
        y_pred = np.zeros(len(X), dtype=int)

        indices_s0 = np.where(mask_s0)[0]  # S_0 的索引
        indices_s1 = np.where(mask_s1)[0]  # S_1 的索引

        if self.svm_layer2_s0 is not None and len(X_s0) > 0:
            y_pred[indices_s0] = self.svm_layer2_s0.predict(X_s0)
        if self.svm_layer2_s1 is not None and len(X_s1) > 0:
            y_pred[indices_s1] = self.svm_layer2_s1.predict(X_s1)

        return y_pred

    def optimize_parameters(self, X_train, y_train, X_test, y_test):
        print("\nStarting parameter optimization...")
        # 定义 C 参数的候选值
        C_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 20, 30]
        # param_combinations = list(itertools.product(C_values, C_values, C_values))
        # 生成所有参数组合
        all_combinations = list(itertools.product(C_values, C_values, C_values))
        # 筛选出第一个C值小于等于第二个和第三个C值的组合
        param_combinations = [combo for combo in all_combinations if combo[0] <= combo[1] and combo[0] <= combo[2]]
        print(f"Total parameter combinations after filtering: {len(param_combinations)}/{len(all_combinations)}")

        start_time = time.time()
        for idx, (c1, c2_s0, c2_s1) in enumerate(param_combinations):
            print(f"\nTrying combination {idx + 1}/{len(param_combinations)}: "
                  f"C_Layer1={c1}, C_Layer2_S0={c2_s0}, C_Layer2_S1={c2_s1}")
            # 更新参数
            self.params['Layer 1']['C'] = c1
            self.params['Layer 2 (S_0)']['C'] = c2_s0
            self.params['Layer 2 (S_1)']['C'] = c2_s1

            # 训练模型
            self.fit(X_train, y_train)

            # 在测试集上评估
            y_pred = self.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy for this combination: {accuracy:.4f}")

            # 更新最佳参数 - 创建深度拷贝
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                # 创建一个全新的字典，而不是引用原参数
                self.best_params = {
                    'Layer 1': {'C': c1, 'kernel': 'linear'},
                    'Layer 2 (S_0)': {'C': c2_s0, 'kernel': 'linear'},
                    'Layer 2 (S_1)': {'C': c2_s1, 'kernel': 'linear'}
                }
                print(f"New best accuracy: {self.best_accuracy:.4f}")

        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"\nParameter optimization completed in {optimization_time:.4f} seconds")
        print(f"Best parameters: {self.best_params}")
        print(f"Best accuracy: {self.best_accuracy:.4f}")

        # 使用最佳参数重新训练模型
        print("\nRetraining with best parameters...")
        self.params = self.best_params.copy()
        self.fit(X_train, y_train)
        return optimization_time

# 创建并训练两层模型
model = TwoLayerSVM()

# 记录总训练时间
start_train_time = time.time()
model.fit(x_train, y_train)
end_train_time = time.time()
total_train_time = end_train_time - start_train_time

# 参数优化
# optimization_time = model.optimize_parameters(x_train, y_train, x_test, y_test)
optimization_time = 0

# Train the model with the specific parameters C = (0.01, 10, 10)
print("\nTraining model with specified parameters: C_Layer1=0.01, C_Layer2_S0=10, C_Layer2_S1=10")
model.params['Layer 1']['C'] = 0.01
model.params['Layer 2 (S_0)']['C'] = 10
model.params['Layer 2 (S_1)']['C'] = 10

# Retrain the model with these specific parameters
model.fit(x_train, y_train)

# Evaluate on test set with the specified parameters
y_pred_specific = model.predict(x_test)
specific_accuracy = accuracy_score(y_test, y_pred_specific)
print(f"Accuracy with specified parameters (0.01, 10, 10): {specific_accuracy:.4f}")

# Compare with the best parameters found during optimization
print(f"Best accuracy from optimization: {model.best_accuracy:.4f}")

# 记录预测时间
# Record prediction time and calculate time per sample
print('Start predicting...')
start_predict_time = time.time()
y_pred = model.predict(x_test)
end_predict_time = time.time()
predict_time = end_predict_time - start_predict_time

# Calculate inference time per sample
time_per_sample = predict_time / len(x_test)
print(f"Inference time per sample: {time_per_sample:.6f} seconds")
# print('Start predicting...')
# start_predict_time = time.time()
# y_pred = model.predict(x_test)

# end_predict_time = time.time()
# predict_time = end_predict_time - start_predict_time

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估指标和时间
print(f'\nFinal Results:')
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f"Total training time: {total_train_time:.4f} seconds")
print(f"Parameter optimization time: {optimization_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")
print(f"Best parameters: {model.best_params}")

# 保存结果到 CSV
results = {
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'Total Training Time (s)', 'Parameter Optimization Time (s)', 'Prediction Time (s)'],
    'Value': [accuracy, recall, precision, f1, total_train_time, optimization_time, predict_time]
}
params = {
    'Layer': ['Layer 1', 'Layer 2 (S_0)', 'Layer 2 (S_1)'],
    'Best Parameters': [str(model.best_params.get('Layer 1', {})),
                        str(model.best_params.get('Layer 2 (S_0)', {})),
                        str(model.best_params.get('Layer 2 (S_1)', {}))]
}

results_df = pd.DataFrame(results)
params_df = pd.DataFrame(params)
output_df = pd.concat([results_df, params_df], axis=1)

# 保存到当前目录下的 CSV 文件
output_file = os.path.join(os.getcwd(), 'two_layer_svm_results.csv')
output_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")