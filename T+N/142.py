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
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 增加一个通道维度
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 Transformer 模型 + 全连接层
class TransformerNet(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, num_classes, dropout_rate):
        super(TransformerNet, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        out = self.transformer_encoder(x)  # Transformer output shape: [batch_size, seq_len, input_dim]
        if len(out.shape) == 3:  # Check if the output is 3D
            out = out[:, -1, :]  # Select the last time step
        return self.fc(out)  # Classification output

# 定义超参数搜索范围
param_distributions = {
    'hidden_dim': randint(64, 256),
    'num_heads': randint(2, 8),
    'num_layers': randint(1, 4),
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
        hidden_dim = params['hidden_dim']
        num_heads = params['num_heads']
        num_layers = params['num_layers']
        dropout_rate = params['dropout_rate']
        learning_rate = params['learning_rate']

        # 初始化模型
        model = TransformerNet(X_train_tensor.size(2), num_heads, hidden_dim, num_layers, 2, dropout_rate)
        model.to(device)

        # 定义优化器和损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        num_epochs = 5  # 每组参数只训练较少轮数以快速评估
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.squeeze(1).to(device), labels.to(device)  # Remove the channel dimension
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
                inputs, labels = inputs.squeeze(1).to(device), labels.to(device)  # Remove the channel dimension
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
hidden_dim = best_params['hidden_dim']
num_heads = best_params['num_heads']
num_layers = best_params['num_layers']
dropout_rate = best_params['dropout_rate']
learning_rate = best_params['learning_rate']
num_epochs = 20  # 训练完整模型时使用较多轮数

model = TransformerNet(X_train_tensor.size(2), num_heads, hidden_dim, num_layers, 2, dropout_rate)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.squeeze(1).to(device), labels.to(device)  # Remove the channel dimension
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
        inputs, labels = inputs.squeeze(1).to(device), labels.to(device)  # Remove the channel dimension
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
model_path = '/Users/han/Charging station/model/142——Transformer_model.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
