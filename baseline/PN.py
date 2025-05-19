import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# 数据路径
DATA_PATH = '/home/user/hrx/CSdata/data.csv'

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据集类
class AnomalyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 数据加载
def load_data():
    column_names = ['ID', 'K1K2驱动信号', '电子锁驱动信号', '急停信号',
                    '门禁信号', 'THDV-M', 'THDI-M', 'label']
    df = pd.read_csv(DATA_PATH, header=None, names=column_names)

    X = df.drop(['ID', 'label'], axis=1).values
    y = df['label'].values

    return X, y, X.shape[1]


# 原型网络模型
class ProtoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 32)
        )

    def forward(self, x):
        return self.encoder(x)

    def compute_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def predict(self, x, prototypes, unique_labels):
        embeddings = self.encoder(x)
        dists = torch.cdist(embeddings, prototypes)
        probs = F.softmax(-dists, dim=1)
        preds = unique_labels[torch.argmin(dists, dim=1)]
        return preds, probs


# 训练函数
def train(model, train_loader, optimizer, epochs=100):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            embeddings = model(batch_x)
            prototypes = model.compute_prototypes(embeddings, batch_y)
            dists = torch.cdist(embeddings, prototypes)
            probs = F.softmax(-dists, dim=1)
            loss = criterion(probs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 评估函数（带预测时间）
def evaluate(model, test_loader, train_loader):
    model.eval()
    all_preds = []
    all_labels = []

    # 计算训练集原型
    with torch.no_grad():
        train_embeddings = []
        train_labels = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            embeddings = model(batch_x)
            train_embeddings.append(embeddings)
            train_labels.append(batch_y)

        train_embeddings = torch.cat(train_embeddings)
        train_labels = torch.cat(train_labels).to(device)
        prototypes = model.compute_prototypes(train_embeddings, train_labels)
        unique_labels = torch.unique(train_labels)

    # 预测测试集并测量时间
    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds, _ = model.predict(batch_x, prototypes, unique_labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    pred_time = time.time() - start_time

    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'pred_time': pred_time
    }
    return metrics


# 交叉验证主程序
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    X, y, input_dim = load_data()
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

    # 设置交叉验证
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # 存储每折结果
    fold_results = []

    # 进行交叉验证
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{k_folds}")

        # 分割数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 创建数据加载器
        train_dataset = AnomalyDataset(X_train, y_train)
        test_dataset = AnomalyDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 初始化模型和优化器
        model = ProtoNet(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        # 训练
        print("Training model...")
        train(model, train_loader, optimizer)

        # 评估
        print("Evaluating model...")
        metrics = evaluate(model, test_loader, train_loader)
        fold_results.append(metrics)

        # 打印单折结果
        print(f"Fold {fold + 1} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Prediction Time: {metrics['pred_time']:.4f} seconds")

    # 计算平均结果
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'pred_time': np.mean([r['pred_time'] for r in fold_results])
    }
    std_metrics = {
        'accuracy': np.std([r['accuracy'] for r in fold_results]),
        'precision': np.std([r['precision'] for r in fold_results]),
        'recall': np.std([r['recall'] for r in fold_results]),
        'f1': np.std([r['f1'] for r in fold_results])
    }

    # 输出平均结果
    print("\nAverage Cross-Validation Results:")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f} (±{std_metrics['accuracy']:.4f})")
    print(f"Precision: {avg_metrics['precision']:.4f} (±{std_metrics['precision']:.4f})")
    print(f"Recall: {avg_metrics['recall']:.4f} (±{std_metrics['recall']:.4f})")
    print(f"F1 Score: {avg_metrics['f1']:.4f} (±{std_metrics['f1']:.4f})")
    print(f"Average Prediction Time: {avg_metrics['pred_time']:.4f} seconds")