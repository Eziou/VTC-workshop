import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import time  # 导入时间模块

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 定义特征提取器（MLP）
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 原型网络
class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_extractor):
        super(PrototypicalNetwork, self).__init__()
        self.feature_extractor = feature_extractor

    def forward(self, support, query, support_labels):
        support_emb = self.feature_extractor(support)
        query_emb = self.feature_extractor(query)

        prototypes = []
        for label in [0, 1]:
            mask = (support_labels == label)
            if mask.sum() > 0:
                prototype = support_emb[mask].mean(dim=0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros(support_emb.shape[1], device=device))
        prototypes = torch.stack(prototypes)

        distances = torch.cdist(query_emb, prototypes, p=2)
        preds = -distances
        return preds

# 3. 训练函数
def train(model, train_loader, optimizer, criterion, epochs=50):
    model.train()
    for epoch in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(data, data, labels)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

# 4. 评估函数
def evaluate(model, support_data, support_labels, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    start_time = time.time()  # 记录预测开始时间
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            support_data = support_data.to(device)
            support_labels = support_labels.to(device)
            preds = model(support_data, data, support_labels)
            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    predict_time = time.time() - start_time  # 计算预测时间
    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return acc, recall, precision, f1, predict_time

# 5. 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path)
    # 列名调整为小写（因为之前保存时可能列名变小写）
    data.columns = data.columns.str.lower()
    # 去掉 'id' 列，保留特征列，假设最后一列是 'label'
    features = data.iloc[:, 1:-1].values  # 去掉 id 和 label 列
    labels = data['label'].values
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

# 6. 主程序
if __name__ == "__main__":
    # 数据集路径
    save_dir = '/home/user/hrx/小样本数据集'
    result_dir = '/home/user/hrx/小样本结果'
    os.makedirs(result_dir, exist_ok=True)

    # 样本量和比例
    sample_sizes = [200, 400, 600, 800, 1000]
    ratios = ['1_1', '1_4', '1_9']  # 比例 1:1, 1:4, 1:9

    # 为每种比例创建结果列表
    results_by_ratio = {ratio: [] for ratio in ratios}

    # 对每种比例分别运行实验
    for ratio in ratios:
        for size in sample_sizes:
            # 加载训练集和测试集
            train_file = os.path.join(save_dir, f'ratio_{ratio}_train_{size}.csv')
            test_file = os.path.join(save_dir, f'ratio_{ratio}_test_{size}.csv')

            # 加载数据
            train_data, train_labels = load_data(train_file)
            test_data, test_labels = load_data(test_file)

            # 创建 DataLoader
            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # 初始化模型
            input_dim = train_data.shape[1]
            feature_extractor = FeatureExtractor(input_dim).to(device)
            model = PrototypicalNetwork(feature_extractor).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # 训练模型
            train(model, train_loader, optimizer, criterion, epochs=50)

            # 评估模型
            acc, recall, precision, f1, predict_time = evaluate(model, train_data, train_labels, test_loader)

            # 保存结果
            results_by_ratio[ratio].append({
                '样本量': size,
                '准确率': acc,
                '召回率': recall,
                '精准率': precision,
                'F1': f1,
                '预测时间': predict_time  # 添加预测时间
            })
            print(f"比例 {ratio}, 样本量 {size} 完成: 准确率={acc:.6f}, 召回率={recall:.6f}, 精准率={precision:.6f}, F1={f1:.6f}, 预测时间={predict_time:.6f}s")

        # 保存该比例的结果到 CSV
        results_df = pd.DataFrame(results_by_ratio[ratio])
        result_file = os.path.join(result_dir, f'proto_ratio_{ratio}_results.csv')
        results_df.to_csv(result_file, index=False)
        print(f"比例 {ratio} 的结果已保存到 {result_file}")

    print("所有实验完成！")
