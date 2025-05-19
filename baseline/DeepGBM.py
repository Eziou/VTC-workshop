import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time

# 数据读取与预处理
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/Users/han/Charging station/data/data.csv", names=col_names)
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
dataset_Y = data[["label"]].values.flatten()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# 转换为torch的张量
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # 确保y_train_tensor是二维的
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)  # 确保y_test_tensor是二维的

# 神经网络模型定义（特征提取部分）
class FeatureNN(nn.Module):
    def __init__(self, input_dim):
        super(FeatureNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 8)
        self.layer5 = nn.Linear(8, 4)
        self.layer6 = nn.Linear(4, 1)  # 输出为二分类结果
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.relu(self.layer5(x))
        x = self.sigmoid(self.layer6(x))
        return x

# 创建并训练神经网络
model = FeatureNN(input_dim=x_train.shape[1])

# 损失函数和优化器
criterion = nn.BCELoss()  # 二分类损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
start_train_time = time.time()
num_epochs = 10  # 设置训练的轮数

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)  # 前向传播
    loss = criterion(outputs, y_train_tensor)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"NN model training time: {train_time:.4f} seconds")

# 获取神经网络的输出作为转换后的特征
model.eval()  # 切换为评估模式
with torch.no_grad():
    x_train_transformed = model(x_train_tensor).numpy()
    x_test_transformed = model(x_test_tensor).numpy()

# 确保 x_train_transformed 和 x_test_transformed 是二维数组
x_train_transformed = x_train_transformed.reshape(-1, 1) if x_train_transformed.ndim == 1 else x_train_transformed
x_test_transformed = x_test_transformed.reshape(-1, 1) if x_test_transformed.ndim == 1 else x_test_transformed

# 创建 LightGBM 数据集，直接使用神经网络的输出作为新的特征
lgb_train = lgb.Dataset(x_train_transformed, y_train)
lgb_eval = lgb.Dataset(x_test_transformed, y_test, reference=lgb_train)

# 设置 LightGBM 超参数
param = {
    'max_depth': 8,
    'num_leaves': 16,
    'learning_rate': 0.4,
    'scale_pos_weight': 1,
    'num_threads': 8,
    'objective': 'binary',
    'bagging_fraction': 1,
    'bagging_freq': 1,
    'min_sum_hessian_in_leaf': 0.01,
    'metric': 'auc',
    'is_unbalance': 'true'
}

# 训练 LightGBM 模型
start_lgb_train_time = time.time()
print('Start training LightGBM...')
gbm = lgb.train(param,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval)

end_lgb_train_time = time.time()
lgb_train_time = end_lgb_train_time - start_lgb_train_time
print(f"LightGBM training time: {lgb_train_time:.4f} seconds")

# 预测
start_predict_time = time.time()
print('Start predicting with LightGBM...')

y_predict_test = gbm.predict(x_test_transformed)
y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_predict_test]

end_predict_time = time.time()
predict_time = end_predict_time - start_predict_time
print(f"Prediction time: {predict_time:.4f} seconds")

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# 打印评估指标
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')
