import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import time  # 导入时间模块

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 定义嵌入模块
class EmbeddingModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):  # 增加隐藏层维度
        super(EmbeddingModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x

# 2. 定义关系模块
class RelationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):  # 增加隐藏层维度
        super(RelationModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# 3. 关系网络
class RelationNetwork:
    def __init__(self, embedding_module, relation_module, lr=0.0001):
        self.embedding_module = embedding_module.to(device)
        self.relation_module = relation_module.to(device)
        self.optimizer = optim.Adam(
            list(self.embedding_module.parameters()) + list(self.relation_module.parameters()),
            lr=lr
        )
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    def compute_prototypes(self, support_data, support_labels):
        embeddings = self.embedding_module(support_data)
        class_0_mask = (support_labels == 0).float().unsqueeze(1)
        class_1_mask = (support_labels == 1).float().unsqueeze(1)
        num_class_0 = torch.sum(class_0_mask) + 1e-6
        num_class_1 = torch.sum(class_1_mask) + 1e-6
        prototype_0 = torch.sum(embeddings * class_0_mask, dim=0) / num_class_0
        prototype_1 = torch.sum(embeddings * class_1_mask, dim=0) / num_class_1
        if num_class_1 < 20:
            noise = torch.randn_like(prototype_1) * 0.01
            prototype_1 = (prototype_1 + noise) / 2
        return prototype_0, prototype_1

    def train(self, train_loader, epochs=300):
        self.embedding_module.train()
        self.relation_module.train()
        for epoch in range(epochs):
            total_loss = 0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                # 移除正类增强，直接使用原始数据
                self.optimizer.zero_grad()
                embeddings = self.embedding_module(data)
                prototype_0, prototype_1 = self.compute_prototypes(data, labels)
                relations_0 = self.relation_module(torch.cat([embeddings, prototype_0.expand_as(embeddings)], dim=1))
                relations_1 = self.relation_module(torch.cat([embeddings, prototype_1.expand_as(embeddings)], dim=1))
                scores = torch.cat([relations_0, relations_1], dim=1)
                probs = F.softmax(scores, dim=1)[:, 1]
                # 使用固定的正类权重
                num_pos = torch.sum(labels == 1).float()
                num_neg = torch.sum(labels == 0).float()
                pos_weight = num_neg / (num_pos + 1e-6)
                criterion = nn.BCELoss(weight=(labels * (pos_weight - 1) + 1).float())
                loss = criterion(probs, labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.embedding_module.parameters()) + list(self.relation_module.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            # 调整学习率
            self.scheduler.step(total_loss / len(train_loader))

    def evaluate(self, support_data, support_labels, test_loader, test_labels):
        self.embedding_module.eval()
        self.relation_module.eval()
        support_data = support_data.to(device)
        support_labels = support_labels.to(device)
        prototype_0, prototype_1 = self.compute_prototypes(support_data, support_labels)
        all_preds = []
        all_labels = []
        start_time = time.time()  # 记录预测开始时间
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                embeddings = self.embedding_module(data)
                relations_0 = self.relation_module(torch.cat([embeddings, prototype_0.expand_as(embeddings)], dim=1))
                relations_1 = self.relation_module(torch.cat([embeddings, prototype_1.expand_as(embeddings)], dim=1))
                scores = torch.cat([relations_0, relations_1], dim=1)
                probs = F.softmax(scores, dim=1)[:, 1]
                predicted = (probs > 0.5).long()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        predict_time = time.time() - start_time  # 计算预测时间
        acc = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        print(f"Pred distribution: Pos={sum(all_preds)}, Neg={len(all_preds) - sum(all_preds)}")
        return acc, recall, precision, f1, predict_time

# 4. 加载数据
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    train_data.columns = train_data.columns.str.lower()
    test_data.columns = test_data.columns.str.lower()
    train_features = train_data.iloc[:, 1:-1].values
    train_labels = train_data['label'].values
    test_features = test_data.iloc[:, 1:-1].values
    test_labels = test_data['label'].values
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    if np.isnan(train_features).any() or np.isinf(train_features).any():
        print(f"Train data in {train_file} contains nan or inf!")
        train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    if np.isnan(test_features).any() or np.isinf(test_features).any():
        print(f"Test data in {test_file} contains nan or inf!")
        test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    return (torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long),
            torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

# 5. 主程序
if __name__ == "__main__":
    save_dir = '/home/user/hrx/小样本数据集'
    result_dir = '/home/user/hrx/小样本结果'
    os.makedirs(result_dir, exist_ok=True)

    sample_sizes = [200, 400, 600, 800, 1000]
    ratios = ['1_1', '1_4', '1_9']

    results_by_ratio = {ratio: [] for ratio in ratios}

    for ratio in ratios:
        for size in sample_sizes:
            train_file = os.path.join(save_dir, f'ratio_{ratio}_train_{size}.csv')
            test_file = os.path.join(save_dir, f'ratio_{ratio}_test_{size}.csv')

            train_data, train_labels, test_data, test_labels = load_data(train_file, test_file)

            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_data)), shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            input_dim = train_data.shape[1]
            embedding_module = EmbeddingModule(input_dim, hidden_dim=256)
            relation_module = RelationModule(input_dim=embedding_module.fc2.out_features * 2, hidden_dim=128)
            relation_net = RelationNetwork(embedding_module, relation_module, lr=0.0001)

            relation_net.train(train_loader, epochs=300)

            acc, recall, precision, f1, predict_time = relation_net.evaluate(train_data, train_labels, test_loader, test_labels)

            results_by_ratio[ratio].append({
                '样本量': size,
                '准确率': acc,
                '召回率': recall,
                '精准率': precision,
                'F1': f1,
                '预测时间': predict_time  # 添加预测时间
            })
            print(f"比例 {ratio}, 样本量 {size} 完成: 准确率={acc:.4f}, 召回率={recall:.4f}, 精准率={precision:.4f}, F1={f1:.4f}, 预测时间={predict_time:.4f}s")

        results_df = pd.DataFrame(results_by_ratio[ratio])
        result_file = os.path.join(result_dir, f'relation_ratio_{ratio}_results.csv')
        results_df.to_csv(result_file, index=False)
        print(f"比例 {ratio} 的结果已保存到 {result_file}")

    print("所有实验完成！")
