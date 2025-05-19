import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import joblib

# 数据加载路径
train_path = '/Users/han/Charging station/data/train_data.csv'
test_path = '/Users/han/Charging station/data/test_data.csv'

# 数据加载函数
def load_data(file_path, has_labels=True):
    try:
        df = pd.read_csv(file_path)
        if has_labels:
            X = df.drop(columns=['id', 'label']).values
            y = df['label'].values
            return X, y
        else:
            X = df.drop(columns=['id']).values
            return X
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None, None

# 加载数据
X_train, y_train = load_data(train_path, has_labels=True)
X_test, y_test = load_data(test_path, has_labels=True)

if X_train is None or X_test is None:
    raise ValueError("Failed to load data.")

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 GRU 模型
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)  # GRU 输出
        return out[:, -1, :]  # 返回最后时间步的特征

# 初始化 GRU 模型
hidden_size = 128
num_layers = 2
dropout_rate = 0.3
input_size = X_train.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gru_model = GRUNet(input_size, hidden_size, num_layers, dropout_rate).to(device)

# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)

# 训练 GRU 模型
num_epochs = 20
gru_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device)
        optimizer.zero_grad()
        features = gru_model(inputs)
        loss = criterion(features, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 提取 GRU 特征
gru_model.eval()
train_features, train_labels = [], []
test_features, test_labels = [], []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1).to(device)
        features = gru_model(inputs).cpu().numpy()
        train_features.append(features)
        train_labels.extend(labels.numpy())

    for inputs, labels in test_loader:
        inputs = inputs.unsqueeze(1).to(device)
        features = gru_model(inputs).cpu().numpy()
        test_features.append(features)
        test_labels.extend(labels.numpy())

train_features = np.vstack(train_features)
test_features = np.vstack(test_features)

# 使用 SVM 进行分类
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(train_features, train_labels)

# 预测测试集
y_pred = svm_model.predict(test_features)
y_pred_proba = svm_model.predict_proba(test_features)[:, 1]  # 类别 1 的概率

# 计算评估指标
accuracy = accuracy_score(test_labels, y_pred)
recall = recall_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')

# 打印混淆矩阵
conf_matrix = confusion_matrix(test_labels, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# 保存 GRU 模型和 SVM 模型
gru_model_path = '/Users/han/Charging station/model/316——GRU_model.pth'
svm_model_path = '/Users/han/Charging station/model/316——SVM_model.pkl'

torch.save(gru_model.state_dict(), gru_model_path)
joblib.dump(svm_model, svm_model_path)

print(f'GRU model saved to {gru_model_path}')
print(f'SVM model saved to {svm_model_path}')
