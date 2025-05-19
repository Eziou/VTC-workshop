import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
import numpy as np
import time

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/Users/han/Charging station/data/data.csv", names=col_names)
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
dataset_Y = data[["label"]].values.flatten()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# 数据预处理：将输入数据转换为Transformer所需的3D形状
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

# 转换为Tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_layers, dropout_rate=0.2):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)

        # 定义Transformer的Encoder部分
        self.transformer = nn.Transformer(
            d_model=hidden_size,  # 隐藏层的维度
            nhead=nhead,  # 多头注意力机制
            num_encoder_layers=num_layers,  # 编码器层数
            dropout=dropout_rate  # Dropout层
        )

        self.fc = nn.Linear(hidden_size, 1)  # 输出层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        x = x.permute(1, 0, 2)  # 调整形状以适应Transformer的输入要求 (seq_len, batch_size, feature)

        # Transformer
        x = self.transformer(x, x)

        # 获取最后一个时间步的输出 (seq_len, batch_size, hidden_size)
        x = x[-1, :, :]

        # 全连接层
        x = self.fc(x)
        return self.sigmoid(x)


# 超参数
input_size = x_train.shape[2]  # 输入特征数
hidden_size = 64  # Transformer隐藏层大小
nhead = 4  # 多头注意力的头数
num_layers = 2  # Transformer编码器层数
dropout_rate = 0.2  # Dropout比率
learning_rate = 0.001
epochs = 50
batch_size = 32

# 创建模型
model = TransformerModel(input_size=input_size, hidden_size=hidden_size, nhead=nhead, num_layers=num_layers,
                         dropout_rate=dropout_rate)
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
start_train_time = time.time()  # 记录训练开始时间
print('Start training...')

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(x_train.size(0))
    for i in range(0, x_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(batch_x)

        # 计算损失
        loss = criterion(outputs, batch_y)

        # 反向传播
        loss.backward()

        # 优化模型参数
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时间

# 预测
start_predict_time = time.time()  # 记录预测开始时间
print('Start predicting...')

model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    y_pred_binary = (y_pred > 0.5).float()

end_predict_time = time.time()  # 记录预测结束时间
predict_time = end_predict_time - start_predict_time  # 计算预测时间

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

# 打印训练和预测时间
print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")
