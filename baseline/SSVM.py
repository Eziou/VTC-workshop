import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
import os

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


# 定义两层 SVM 模型类
class TwoLayerSVM:
    def __init__(self):
        # 第一层和第二层的 SVM（初始为空，待优化后赋值）
        self.svm_layer1 = None
        self.svm_layer2_s0 = None
        self.svm_layer2_s1 = None
        # 记录最佳参数
        self.best_params = {}

    def optimize_svm(self, X, y, layer_name):
        # 定义不同核函数的参数网格
        param_grid = [
            {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
            {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]},
            {'kernel': ['poly'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01], 'degree': [2, 3],
             'coef0': [0, 1]},
            {'kernel': ['sigmoid'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01], 'coef0': [0, 1]}
        ]
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # 记录优化时间
        start_time = time.time()
        grid_search.fit(X, y)
        end_time = time.time()

        # 保存最佳参数
        self.best_params[layer_name] = grid_search.best_params_
        print(f"{layer_name} - Best parameters: {grid_search.best_params_}")
        print(f"{layer_name} optimization time: {end_time - start_time:.4f} seconds")

        return grid_search.best_estimator_

    def fit(self, X, y):
        # 优化并训练第一层
        print('Optimizing and training Layer 1...')
        self.svm_layer1 = self.optimize_svm(X, y, 'Layer 1')

        # 第一层预测训练集标签
        labels_layer1 = self.svm_layer1.predict(X)

        # 分割数据为 S_0 和 S_1
        mask_s0 = labels_layer1 == 0  # 子集 S_0
        mask_s1 = labels_layer1 == 1  # 子集 S_1

        X_s0, y_s0 = X[mask_s0], y[mask_s0]
        X_s1, y_s1 = X[mask_s1], y[mask_s1]

        # 优化并训练第二层 S_0
        print('Optimizing and training Layer 2 (Subset S_0)...')
        if len(X_s0) > 0 and len(np.unique(y_s0)) > 1:
            self.svm_layer2_s0 = self.optimize_svm(X_s0, y_s0, 'Layer 2 (S_0)')
        else:
            print("Subset S_0 has insufficient data or classes for training.")
            self.svm_layer2_s0 = None

        # 优化并训练第二层 S_1
        print('Optimizing and training Layer 2 (Subset S_1)...')
        if len(X_s1) > 0 and len(np.unique(y_s1)) > 1:
            self.svm_layer2_s1 = self.optimize_svm(X_s1, y_s1, 'Layer 2 (S_1)')
        else:
            print("Subset S_1 has insufficient data or classes for training.")
            self.svm_layer2_s1 = None

    def predict(self, X):
        # 第一层预测
        labels_layer1 = self.svm_layer1.predict(X)

        # 初始化最终预测结果
        y_pred = np.zeros(len(X), dtype=int)

        # 第二层预测：一次性处理 S_0 和 S_1
        mask_s0 = labels_layer1 == 0
        mask_s1 = labels_layer1 == 1

        if self.svm_layer2_s0 is not None and len(X[mask_s0]) > 0:
            y_pred[mask_s0] = self.svm_layer2_s0.predict(X[mask_s0])
        if self.svm_layer2_s1 is not None and len(X[mask_s1]) > 0:
            y_pred[mask_s1] = self.svm_layer2_s1.predict(X[mask_s1])

        return y_pred


# 创建并训练两层模型
model = TwoLayerSVM()

# 记录总训练时间
start_train_time = time.time()
model.fit(x_train, y_train)
end_train_time = time.time()
total_train_time = end_train_time - start_train_time

# 记录预测时间
print('Start predicting...')
start_predict_time = time.time()
y_pred = model.predict(x_test)
end_predict_time = time.time()
predict_time = end_predict_time - start_predict_time

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
print(f"Prediction time: {predict_time:.4f} seconds")
print(f"Best parameters: {model.best_params}")

# 保存结果到 CSV
results = {
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'Total Training Time (s)', 'Prediction Time (s)'],
    'Value': [accuracy, recall, precision, f1, total_train_time, predict_time]
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