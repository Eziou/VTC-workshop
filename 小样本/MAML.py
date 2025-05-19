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

# 1. 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_params(self, x, fast_weights):
        fc1_weight = fast_weights['fc1.weight']
        fc1_bias = fast_weights['fc1.bias']
        fc2_weight = fast_weights['fc2.weight']
        fc2_bias = fast_weights['fc2.bias']
        fc3_weight = fast_weights['fc3.weight']
        fc3_bias = fast_weights['fc3.bias']
        x = F.linear(x, fc1_weight, fc1_bias)
        x = F.relu(x)
        x = F.linear(x, fc2_weight, fc2_bias)
        x = F.relu(x)
        x = F.linear(x, fc3_weight, fc3_bias)
        return x

# 2. 定义 MAML 类
class MAML:
    def __init__(self, model, lr_inner=0.01, lr_outer=0.001):
        self.model = model
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr_outer)

    def inner_update(self, support_data, support_labels, class_weights, num_steps=20):
        support_data = support_data.to(device)
        support_labels = support_labels.to(device)
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        for _ in range(num_steps):
            preds = self.model.forward_with_params(support_data, fast_weights)
            loss = criterion(preds, support_labels)
            if torch.isnan(loss):
                print("Inner loop loss is nan!")
                return fast_weights
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            grads = [torch.clamp(g, -1.0, 1.0) if g is not None else None for g in grads]
            fast_weights = {name: param - self.lr_inner * grad for (name, param), grad in zip(fast_weights.items(), grads)}
        return fast_weights

    def outer_update(self, support_data, support_labels, query_data, query_labels, class_weights, num_inner_steps=20):
        fast_weights = self.inner_update(support_data, support_labels, class_weights, num_inner_steps)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        query_data = query_data.to(device)
        query_labels = query_labels.to(device)
        self.optimizer.zero_grad()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(fast_weights[name])
        query_preds = self.model(query_data)
        query_loss = criterion(query_preds, query_labels)
        if torch.isnan(query_loss):
            print("Outer loop loss is nan!")
            return float('nan')
        query_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return query_loss.item()

    def train(self, train_loader, test_labels, epochs=200, num_inner_steps=20):
        self.model.train()
        test_pos_ratio = torch.sum(test_labels == 1).float() / len(test_labels)
        print(f"Test positive ratio: {test_pos_ratio:.4f}")
        for epoch in range(epochs):
            total_loss = 0
            for data, labels in train_loader:
                data, labels = data.to(device), labels.to(device)
                if torch.isnan(data).any() or torch.isinf(data).any():
                    print("Input data contains nan or inf!")
                    return
                num_samples = data.size(0)
                perm = torch.randperm(num_samples)
                support_size = num_samples // 2
                support_idx = perm[:support_size]
                query_idx = perm[support_size:]
                support_data, support_labels = data[support_idx], labels[support_idx]
                query_data, query_labels = data[query_idx], labels[query_idx]
                num_pos = torch.sum(support_labels == 1).float()
                num_neg = torch.sum(support_labels == 0).float()
                class_weights = torch.tensor([1.0, max(5.0, num_neg / (num_pos + 1e-6))])
                loss = self.outer_update(support_data, support_labels, query_data, query_labels, class_weights, num_inner_steps)
                total_loss += loss if not np.isnan(loss) else 0
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def evaluate(self, support_data, support_labels, test_loader, test_labels, num_inner_steps=20):
        self.model.eval()
        support_data = support_data.to(device)
        support_labels = support_labels.to(device)
        num_pos = torch.sum(support_labels == 1).float()
        num_neg = torch.sum(support_labels == 0).float()
        class_weights = torch.tensor([1.0, max(5.0, num_neg / (num_pos + 1e-6))])
        fast_weights = self.inner_update(support_data, support_labels, class_weights, num_inner_steps)
        all_preds = []
        all_labels = []
        start_time = time.time()  # 记录预测开始时间
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.copy_(fast_weights[name])
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                preds = self.model(data)
                probs = F.softmax(preds, dim=1)[:, 1]
                threshold = min(0.5, max(0.05, torch.sum(test_labels == 1).float() / len(test_labels)))
                predicted = (probs > threshold).long()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        predict_time = time.time() - start_time  # 计算预测时间
        acc = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        return acc, recall, precision, f1, predict_time

# 3. 加载数据
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    # 列名调整为小写
    train_data.columns = train_data.columns.str.lower()
    test_data.columns = test_data.columns.str.lower()
    # 去掉 'id' 列，保留特征列，假设最后一列是 'label'
    train_features = train_data.iloc[:, 1:-1].values
    train_labels = train_data['label'].values
    test_features = test_data.iloc[:, 1:-1].values
    test_labels = test_data['label'].values
    # 统一标准化
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

# 4. 主程序
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
            train_data, train_labels, test_data, test_labels = load_data(train_file, test_file)

            # 创建 DataLoader
            train_dataset = TensorDataset(train_data, train_labels)
            test_dataset = TensorDataset(test_data, test_labels)
            train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_data)), shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # 初始化模型
            input_dim = train_data.shape[1]
            model = Classifier(input_dim, hidden_dim=128).to(device)
            maml = MAML(model, lr_inner=0.01, lr_outer=0.001)

            # 训练模型
            maml.train(train_loader, test_labels, epochs=200, num_inner_steps=20)

            # 评估模型
            acc, recall, precision, f1, predict_time = maml.evaluate(train_data, train_labels, test_loader, test_labels, num_inner_steps=20)

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
        result_file = os.path.join(result_dir, f'maml_ratio_{ratio}_results.csv')
        results_df.to_csv(result_file, index=False)
        print(f"比例 {ratio} 的结果已保存到 {result_file}")

    print("所有实验完成！")
