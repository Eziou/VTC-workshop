import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import joblib


class ExtremeLearningMachine:
    def __init__(self, n_hidden=100, activation='sigmoid'):
        self.n_hidden = n_hidden
        self.activation = activation

    def radbas(self, x):
        return np.exp(-np.square(x))

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)  # 防止数值溢出
        return 1 / (1 + np.exp(-x))

    def sine(self, x):
        return np.sin(x)

    def hardlim(self, x):
        return np.where(x >= 0, 1, 0)

    def tribas(self, x):
        return np.maximum(1 - np.abs(x), 0)

    def fit(self, X, y):
        # 获取激活函数
        if self.activation == 'radbas':
            act_func = self.radbas
        elif self.activation == 'sigmoid':
            act_func = self.sigmoid
        elif self.activation == 'sine':
            act_func = self.sine
        elif self.activation == 'hardlim':
            act_func = self.hardlim
        elif self.activation == 'tribas':
            act_func = self.tribas
        else:
            raise ValueError("Unsupported activation function")

        # 初始化权重和偏置
        self.W_input = np.random.randn(X.shape[1], self.n_hidden)
        self.b_hidden = np.random.randn(self.n_hidden)

        # 计算隐层输出
        H = act_func(np.dot(X, self.W_input) + self.b_hidden)

        # 使用最小二乘法求解输出权重
        self.W_output = np.linalg.pinv(H).dot(y.reshape(-1, 1)).flatten()

    def predict_proba(self, X):
        # 获取激活函数
        if self.activation == 'radbas':
            act_func = self.radbas
        elif self.activation == 'sigmoid':
            act_func = self.sigmoid
        elif self.activation == 'sine':
            act_func = self.sine
        elif self.activation == 'hardlim':
            act_func = self.hardlim
        elif self.activation == 'tribas':
            act_func = self.tribas
        else:
            raise ValueError("Unsupported activation function")

        H = act_func(np.dot(X, self.W_input) + self.b_hidden)
        return np.dot(H, self.W_output)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, 0)


# 数据加载函数
def load_data(file_path, has_labels=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if has_labels:
        X = df.drop(columns=['id', 'label']).values
        y = df['label'].values
        return X, y, df  # 返回原始 DataFrame 用于保存未识别异常
    else:
        X = df.drop(columns=['id']).values
        return X, None, df


# 加载数据
train_path = '/Users/han/Charging station/data/train_data.csv'
test_path = '/Users/han/Charging station/data/test_data.csv'
output_path = '/Users/han/Charging station/data base/ELM_unrecognized_anomalies.csv'

X_train, y_train, _ = load_data(train_path, has_labels=True)
X_test, y_test, test_df = load_data(test_path, has_labels=True)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 检查数据完整性
if np.isnan(X_train).any() or np.isnan(X_test).any() or np.isinf(X_train).any() or np.isinf(X_test).any():
    raise ValueError("Data contains invalid values.")

# 定义激活函数和隐藏层数参数
activation_functions = ['radbas', 'sigmoid', 'sine', 'hardlim', 'tribas']
hidden_layer_sizes = [10, 20, 40, 60, 80]

# 保存所有结果
results = []

for activation in activation_functions:
    for n_hidden in hidden_layer_sizes:
        elm = ExtremeLearningMachine(n_hidden=n_hidden, activation=activation)
        elm.fit(X_train, y_train)
        y_pred = elm.predict(X_test)

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # 保存结果
        results.append({
            'activation': activation,
            'n_hidden': n_hidden,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })

        print(f'Activation: {activation}, Hidden Layers: {n_hidden}')
        print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1: {f1}')
        print('-' * 50)

        # 筛选未正确分类的异常
        unrecognized = test_df[(y_test == 1) & (y_pred == 0)]  # 实际异常但预测为正常
        if not unrecognized.empty:
            print(f'Found {len(unrecognized)} unrecognized anomalies.')
            unrecognized.to_csv(output_path, index=False)
            print(f'Unrecognized anomalies saved to {output_path}')

# 保存最佳模型和结果
best_result = max(results, key=lambda x: x['f1'])
print(f'Best Model: {best_result}')
joblib.dump(best_result, '/Users/han/Charging station/baseline-model/ELM_model.pkl')
