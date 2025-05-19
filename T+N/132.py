import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import ParameterSampler
from scipy.stats import randint, uniform

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

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 为 CNN 添加通道维度
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)  # 为 CNN 添加通道维度
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 CNN 模型
class CNNNet(nn.Module):
    def __init__(self, input_channels, input_length, num_classes, dropout_rate):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)

        # 动态计算全连接层的输入维度
        self.fc_input_dim = 128 * (input_length // 4)
        self.fc = nn.Linear(self.fc_input_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 定义超参数搜索范围
param_distributions = {
    'dropout_rate': uniform(0.1, 0.5),
    'learning_rate': uniform(0.0001, 0.01)
}
num_samples = 5  # 随机采样次数

# 超参数优化函数
def random_search(X_train_tensor, y_train_tensor, param_distributions, num_samples):
    best_params = None
    best_f1 = 0.0
    param_sampler = ParameterSampler(param_distributions, n_iter=num_samples, random_state=42)

    for params in param_sampler:
        print(f"Testing parameters: {params}")
        dropout_rate = params['dropout_rate']
        learning_rate = params['learning_rate']

        # 初始化模型
        model = CNNNet(1, X_train_tensor.size(2), 2, dropout_rate)
        model.to(device)

        # 定义优化器和损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        num_epochs = 5  # 每组参数只训练较少轮数以快速评估
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 测试模型
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # 计算评估指标
        f1 = f1_score(all_labels, all_preds)
        print(f"F1 Score for current params: {f1:.4f}")

        # 更新最佳参数
        if f1 > best_f1:
            best_f1 = f1
            best_params = params

    print(f"Best parameters: {best_params}, Best F1 Score: {best_f1:.4f}")
    return best_params

# 搜索最佳超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_params = random_search(X_train_tensor, y_train_tensor, param_distributions, num_samples)

# 使用最佳参数训练最终模型
dropout_rate = best_params['dropout_rate']
learning_rate = best_params['learning_rate']
num_epochs = 20  # 训练完整模型时使用较多轮数

model = CNNNet(1, X_train_tensor.size(2), 2, dropout_rate)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 测试最终模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')

# 打印混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(conf_matrix)

# 保存最终模型
model_path = '/Users/han/Charging station/model/132——CNN_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
