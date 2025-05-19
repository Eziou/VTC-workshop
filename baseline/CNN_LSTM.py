import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import optuna
import time

# 数据加载路径
train_path = '/Users/han/Charging station/data/train_data.csv'
test_path = '/Users/han/Charging station/data/test_data.csv'

# 加载数据
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 提取数据
X_train = df_train.drop(columns=['id', 'label']).values
y_train = df_train['label'].values
X_test = df_test.drop(columns=['id', 'label']).values
y_test = df_test['label'].values
ids_test = df_test['id'].values

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 检查数据
if np.isnan(X_train).any() or np.isnan(X_test).any():
    raise ValueError("Data contains NaN values.")
if np.isinf(X_train).any() or np.isinf(X_test).any():
    raise ValueError("Data contains infinite values.")

# 创建序列数据
time_steps = 10

def create_sequences(data, labels, ids, time_steps):
    sequences = []
    seq_labels = []
    seq_ids = []
    for i in range(len(data) - time_steps + 1):
        seq = data[i:i + time_steps]
        label = labels[i + time_steps - 1]
        sequences.append(seq)
        seq_labels.append(label)
        if ids is not None:
            seq_ids.append(ids[i + time_steps - 1])
    if ids is not None:
        return np.array(sequences), np.array(seq_labels), np.array(seq_ids)
    else:
        return np.array(sequences), np.array(seq_labels), None

X_train_seq, y_train_seq, _ = create_sequences(X_train, y_train, None, time_steps)
X_test_seq, y_test_seq, ids_test_seq = create_sequences(X_test, y_test, ids_test, time_steps)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.long)

# 数据加载器
batch_size = 128
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义 CNN-LSTM 模型
class CNN_LSTM(nn.Module):
    def __init__(self, num_features, hidden_dim, layer_dim, output_dim, kernel_size=3):
        super(CNN_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.num_features = num_features
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # CNN 层
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=6, kernel_size=kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(num_features=6)
        self.dropout1 = nn.Dropout(0.5)

        # LSTM 层
        self.lstm = nn.LSTM(num_features, hidden_dim, layer_dim, batch_first=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 改变形状为 (batch_size, num_features, time_steps)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # 改变形状回 (batch_size, time_steps, channels)

        # LSTM 前向传播
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # 获取最后时间步的输出
        out = self.fc(out[:, -1, :])
        return out

# 定义评估函数
def evaluate(model, loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, target in loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(target.cpu().numpy())
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# 使用 Optuna 进行超参数搜索
def objective(trial):
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    layer_dim = trial.suggest_int('layer_dim', 1, 3)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    model = CNN_LSTM(num_features=X_train.shape[1], hidden_dim=hidden_dim, layer_dim=layer_dim, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 训练过程
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # 评估模型
    accuracy = evaluate(model, test_loader)
    return accuracy

# 使用 Optuna 进行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 输出最优参数
print(f"Best parameters found: {study.best_params}")

# 使用最优参数训练模型
best_params = study.best_params
model = CNN_LSTM(num_features=X_train.shape[1], hidden_dim=best_params['hidden_dim'], layer_dim=best_params['layer_dim'], output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 20
model.train()
start_train_time = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')
end_train_time = time.time()
train_time = end_train_time - start_train_time

# 测试模型并保存未正确分类的样本
model.eval()
y_pred = []
incorrect_indices = []
start_idx = 0

# 记录预测开始时间
start_predict_time = time.time()  # 记录开始预测的时间
print('Start predicting...')

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        incorrect_batch_indices = np.where(predicted.cpu().numpy() != target.cpu().numpy())[0]
        incorrect_indices.extend(start_idx + incorrect_batch_indices)
        start_idx += len(target)

# 记录预测结束时间
end_predict_time = time.time()  # 记录预测结束的时间
predict_time = end_predict_time - start_predict_time  # 计算预测时间

# 保存未正确分类的样本
incorrect_ids = ids_test_seq[incorrect_indices]
incorrect_samples = df_test[df_test['id'].isin(incorrect_ids)]
save_path = '/Users/han/Charging station/data base/CNN_LSTM_incorrect_samples.csv'
incorrect_samples.to_csv(save_path, index=False)
print(f"Incorrect samples saved to {save_path}")

# 计算评估指标
accuracy = accuracy_score(y_test_seq, y_pred)
recall = recall_score(y_test_seq, y_pred)
precision = precision_score(y_test_seq, y_pred)
f1 = f1_score(y_test_seq, y_pred)

# 打印评估指标
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

# 打印训练和预测时间
print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")  # 打印预测时间

# 保存模型
model_save_path = '/Users/han/Charging station/baseline-model/CNN_LSTM_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
