import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

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

# 定义 GRU + 全连接层模型
class GRUNetWithFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, num_classes):
        super(GRUNetWithFC, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]  # 取最后时间步的输出 [batch_size, hidden_size]
        out = self.fc(out)  # 全连接层分类
        return out

# 初始化模型
hidden_size = 128
num_layers = 2
dropout_rate = 0.3
learning_rate = 0.001
num_epochs = 20
num_classes = 2

model = GRUNetWithFC(X_train_tensor.size(1), hidden_size, num_layers, dropout_rate, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 对于多类分类问题
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device)  # 添加时间维度
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 测试模型
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.unsqueeze(1).to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f'Accuracy: {accuracy:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Precision: {precision:.2f}')
print(f'F1 Score: {f1:.2f}')

# 打印混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)
print('Confusion Matrix:')
print(conf_matrix)

# 保存最终模型
model_path = '/Users/han/Charging station/model/212——GRU_with_FC.pth'
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
