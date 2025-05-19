import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import time  # 导入 time 库

# 定义路径
train_path = '/Users/han/Charging station/data/train_data.csv'
test_path = '/Users/han/Charging station/data/test_data.csv'
incorrect_save_path = '/Users/han/Charging station/data base/CNN_incorrect_samples.csv'
model_save_path = '/Users/han/Charging station/baseline-model/CNN_model.pth'

# 确保保存目录存在
os.makedirs(os.path.dirname(incorrect_save_path), exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# 加载数据
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 数据预处理：去除 id 和 label 列
X_train = df_train.drop(columns=['id', 'label']).values
y_train = df_train['label'].values
X_test = df_test.drop(columns=['id', 'label']).values
y_test = df_test['label'].values
ids_test = df_test['id'].values  # 保留 id 列

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 确保输入特征的维度为 CNN 设计的完全平方数
input_dim = X_train.shape[1]
feature_map_size = int(np.ceil(np.sqrt(input_dim)))
padded_input_dim = feature_map_size * feature_map_size

# 填充特征以满足 CNN 的输入维度
X_train_padded = np.pad(X_train, ((0, 0), (0, padded_input_dim - input_dim)), mode='constant')
X_test_padded = np.pad(X_test, ((0, 0), (0, padded_input_dim - input_dim)), mode='constant')

# 重塑为 CNN 所需的形状
X_train_reshaped = X_train_padded.reshape(-1, 1, feature_map_size, feature_map_size)
X_test_reshaped = X_test_padded.reshape(-1, 1, feature_map_size, feature_map_size)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 数据加载器
batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self, input_channels, num_classes, feature_map_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # 计算全连接层输入大小
        after_pool = feature_map_size // 2
        if after_pool < 1:
            raise ValueError(f"Feature map size too small after pooling: {after_pool}. Adjust the CNN structure.")
        fc_input_size = 32 * after_pool * after_pool

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 参数优化
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# 定义评估指标
scorer = make_scorer(f1_score)

# 定义参数网格
param_grid = {
    'lr': [0.001, 0.0001],
    'num_epochs': [10, 20],
    'batch_size': [32, 64]
}

# 定义训练函数
def train_and_evaluate(lr, num_epochs, batch_size):
    # 初始化模型
    input_channels = 1  # 定义 input_channels 变量
    num_classes = 2  # 定义 num_classes 变量
    model = CNN(input_channels, num_classes, feature_map_size)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    model.train()
    start_train_time = time.time()  # 记录开始训练的时间
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    end_train_time = time.time()  # 记录训练结束的时间
    train_time = end_train_time - start_train_time  # 计算训练时间

    # 测试模型并保存未正确分类的样本
    model.eval()
    y_pred = []
    incorrect_indices = []
    start_predict_time = time.time()  # 记录开始预测的时间
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            incorrect_indices.extend(np.where(predicted.cpu().numpy() != target.cpu().numpy())[0])
    end_predict_time = time.time()  # 记录预测结束的时间
    predict_time = end_predict_time - start_predict_time  # 计算预测时间

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return f1, train_time, predict_time

# 使用 GridSearchCV 进行参数搜索
def grid_search(param_grid):
    best_f1 = 0
    best_params = {}
    best_train_time = 0
    best_predict_time = 0

    for lr in param_grid['lr']:
        for num_epochs in param_grid['num_epochs']:
            for batch_size in param_grid['batch_size']:
                print(f"Training with lr={lr}, num_epochs={num_epochs}, batch_size={batch_size}")
                f1, train_time, predict_time = train_and_evaluate(lr, num_epochs, batch_size)
                print(f"F1 Score: {f1:.4f}, Training time: {train_time:.4f} seconds, Prediction time: {predict_time:.4f} seconds")

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'lr': lr, 'num_epochs': num_epochs, 'batch_size': batch_size}
                    best_train_time = train_time
                    best_predict_time = predict_time

    return best_params, best_f1, best_train_time, best_predict_time

# 执行网格搜索
best_params, best_f1, best_train_time, best_predict_time = grid_search(param_grid)
print(f"Best Parameters: {best_params}, Best F1: {best_f1:.4f}, Best Training time: {best_train_time:.4f} seconds, Best Prediction time: {best_predict_time:.4f} seconds")

# 使用最佳参数重新训练模型
input_channels = 1  # 单通道输入
num_classes = 2  # 二分类问题
model = CNN(input_channels, num_classes, feature_map_size)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()

# 训练模型
model.train()
start_train_time = time.time()  # 记录开始训练的时间
for epoch in range(best_params['num_epochs']):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
end_train_time = time.time()  # 记录训练结束的时间
train_time = end_train_time - start_train_time  # 计算训练时间

# 测试模型并保存未正确分类的样本
model.eval()
y_pred = []
incorrect_indices = []
start_predict_time = time.time()  # 记录开始预测的时间
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        incorrect_indices.extend(np.where(predicted.cpu().numpy() != target.cpu().numpy())[0])
end_predict_time = time.time()  # 记录预测结束的时间
predict_time = end_predict_time - start_predict_time  # 计算预测时间

# 打印未正确分类的样本数量
print(f"Number of incorrectly classified samples: {len(incorrect_indices)}")

# 保存未正确分类的样本
incorrect_ids = ids_test[incorrect_indices]
incorrect_samples = df_test[df_test['id'].isin(incorrect_ids)]
incorrect_samples.to_csv(incorrect_save_path, index=False)
print(f"Incorrect samples saved to {incorrect_save_path}")

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估指标
print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')

# 打印训练和预测时间
print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")

# 保存模型
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")