import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import time
import os

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/home/user/hrx/完整版/预训练/data/data.csv", names=col_names)

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
    def __init__(self, epochs=10):
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
        self.epochs = epochs

    def train_svm(self, X, y, layer_name, sample_weight=None):
        # 根据层级设置参数
        params = self.params[layer_name]
        svm = SVC(kernel='linear', C=params['C'], random_state=42)
        start_time = time.time()
        svm.fit(X, y, sample_weight=sample_weight)
        end_time = time.time()
        print(f"{layer_name} - Using parameters: {params}")
        print(f"{layer_name} training time: {end_time - start_time:.4f} seconds")
        return svm

    def fit(self, X, y, sample_weight=None):
        # 训练第一层
        print('Training Layer 1...')
        self.svm_layer1 = self.train_svm(X, y, 'Layer 1', sample_weight)

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
            self.svm_layer2_s0 = self.train_svm(X_s0, y_s0, 'Layer 2 (S_0)',
                                               sample_weight[mask_s0] if sample_weight is not None else None)
        else:
            print("Subset S_0 has insufficient data or classes for training.")
            self.svm_layer2_s0 = None

        # 训练第二层 S_1
        print('Training Layer 2 (Subset S_1)...')
        if len(X_s1) > 0 and len(np.unique(y_s1)) > 1:
            self.svm_layer2_s1 = self.train_svm(X_s1, y_s1, 'Layer 2 (S_1)',
                                               sample_weight[mask_s1] if sample_weight is not None else None)
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

    def optimize_with_weights(self, X_train, y_train, X_test, y_test):
        print("\nStarting optimization with sample weights...")
        start_time = time.time()

        # 初始样本权重
        sample_weights = np.ones(len(X_train))

        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            # 训练模型
            self.fit(X_train, y_train, sample_weight=sample_weights)

            # 在测试集上评估
            y_pred = self.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy at epoch {epoch + 1}: {accuracy:.4f}")

            # 更新最佳参数
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_params = self.params.copy()
                print(f"New best accuracy: {self.best_accuracy:.4f}")

            # 计算训练集上的错误
            y_train_pred = self.predict(X_train)
            errors = (y_train_pred != y_train)

            # 更新样本权重：错误分类的样本权重增加
            sample_weights[errors] = 2.0  # 错误分类的样本权重设为 2.0
            sample_weights[~errors] = 1.0  # 正确分类的样本权重设为 1.0
            print(f"Number of misclassified samples: {np.sum(errors)}")

        end_time = time.time()
        optimization_time = end_time - start_time
        print(f"\nOptimization with weights completed in {optimization_time:.4f} seconds")
        print(f"Best accuracy: {self.best_accuracy:.4f}")

        # 使用最佳参数重新训练模型
        print("\nRetraining with best parameters...")
        self.fit(X_train, y_train, sample_weight=sample_weights)
        return optimization_time

# 创建并训练两层模型
model = TwoLayerSVM(epochs=10)

# 记录总训练时间
start_train_time = time.time()
model.fit(x_train, y_train)
end_train_time = time.time()
total_train_time = end_train_time - start_train_time

# 使用样本权重优化
optimization_time = model.optimize_with_weights(x_train, y_train, x_test, y_test)

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
print(f"Optimization time: {optimization_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")
print(f"Best parameters: {model.best_params}")

# 保存结果到 CSV
results = {
    'Metric': ['Accuracy', 'Recall', 'Precision', 'F1 Score', 'Total Training Time (s)', 'Optimization Time (s)', 'Prediction Time (s)'],
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
output_file = os.path.join(os.getcwd(), '权重two_layer_svm_results.csv')
output_df.to_csv(output_file, index=False)
print(f"Results saved to: {output_file}")