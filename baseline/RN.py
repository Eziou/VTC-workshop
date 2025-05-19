import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
from tqdm import tqdm

# 数据路径
DATA_PATH = '/home/user/hrx/CSdata/data.csv'
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


# 数据加载与预处理
def load_data():
    column_names = ['ID', 'K1K2驱动信号', '电子锁驱动信号', '急停信号',
                    '门禁信号', 'THDV-M', 'THDI-M', 'label']
    df = pd.read_csv(DATA_PATH, header=None, names=column_names)

    X = df.drop(['ID', 'label'], axis=1).values
    y = df['label'].values

    # 标准化特征
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, X.shape[1]


# 特征嵌入网络
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):  # 增加隐藏层维度
        super(FeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 64),  # 增加嵌入维度
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

    def forward(self, x):
        return self.encoder(x)


# 关系网络
class RelationNetwork(nn.Module):
    def __init__(self, input_dim=128):  # 两个64维嵌入拼接
        super(RelationNetwork, self).__init__()
        self.relation = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)  # 输出logits，不用Sigmoid
        )

    def forward(self, x):
        return self.relation(x)


# 完整模型
class RelationNet(nn.Module):
    def __init__(self, input_dim):
        super(RelationNet, self).__init__()
        self.feature_encoder = FeatureEncoder(input_dim)
        self.relation_net = RelationNetwork(input_dim=128)

    def compute_prototypes(self, embeddings, labels):
        unique_labels = torch.unique(labels)
        prototypes = []
        for label in unique_labels:
            class_embeddings = embeddings[labels == label]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes), unique_labels

    def forward(self, x, prototypes):
        embeddings = self.feature_encoder(x)
        relations = []
        for proto in prototypes:
            combined = torch.cat((embeddings, proto.repeat(embeddings.size(0), 1)), dim=1)
            relation_score = self.relation_net(combined)
            relations.append(relation_score)
        return torch.cat(relations, dim=1)  # [batch_size, num_classes]

    def predict(self, x, prototypes, unique_labels):
        probs = self.forward(x, prototypes)
        preds = unique_labels[torch.argmax(probs, dim=1)]
        return preds, probs


# 训练函数
def train(model, train_loader, optimizer, epochs=100):  # 增加epochs
    model.train()
    criterion = nn.CrossEntropyLoss()

    for _ in tqdm(range(epochs), desc="Training"):
        # 动态计算原型
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                embeddings = model.feature_encoder(batch_x)
                all_embeddings.append(embeddings)
                all_labels.append(batch_y)
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels).to(device)
        prototypes, unique_labels = model.compute_prototypes(all_embeddings, all_labels)

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            probs = model(batch_x, prototypes)
            loss = criterion(probs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 评估函数
def evaluate(model, loader, prototypes, unique_labels, set_name="Test"):
    model.eval()
    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds, _ = model.predict(batch_x, prototypes, unique_labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    pred_time = time.time() - start_time

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'pred_time': pred_time
    }
    return metrics


# 主程序
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    X, y, input_dim = load_data()
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

    # 设置交叉验证
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    final_train_metrics = None
    final_val_metrics = None
    final_test_metrics = None
    total_train_time = 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        train_idx_sub, val_idx_sub = train_test_split(
            np.arange(len(X_train_full)), test_size=0.2, stratify=y_train_full, random_state=42
        )
        X_train, X_val = X_train_full[train_idx_sub], X_train_full[val_idx_sub]
        y_train, y_val = y_train_full[train_idx_sub], y_train_full[val_idx_sub]

        train_dataset = AnomalyDataset(X_train, y_train)
        val_dataset = AnomalyDataset(X_val, y_val)
        test_dataset = AnomalyDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 增大batch_size
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        model = RelationNet(input_dim=input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)  # 提高学习率

        start_time = time.time()
        train(model, train_loader, optimizer)
        fold_train_time = time.time() - start_time
        total_train_time += fold_train_time

        # 计算测试原型
        with torch.no_grad():
            train_embeddings = model.feature_encoder(torch.FloatTensor(X_train).to(device))
            prototypes, unique_labels = model.compute_prototypes(train_embeddings, torch.LongTensor(y_train).to(device))

        if fold == k_folds - 1:
            final_train_metrics = evaluate(model, train_loader, prototypes, unique_labels, "Train")
            final_val_metrics = evaluate(model, val_loader, prototypes, unique_labels, "Validation")
            final_test_metrics = evaluate(model, test_loader, prototypes, unique_labels, "Test")

    # 打印最终结果
    print("\n===== Final Results =====")
    print("Train Metrics:")
    print(f"Accuracy: {final_train_metrics['accuracy']:.4f}")
    print(f"Precision: {final_train_metrics['precision']:.4f}")
    print(f"Recall: {final_train_metrics['recall']:.4f}")
    print(f"F1 Score: {final_train_metrics['f1']:.4f}")
    print(f"Prediction Time: {final_train_metrics['pred_time']:.4f} seconds")

    print("\nValidation Metrics:")
    print(f"Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"Precision: {final_val_metrics['precision']:.4f}")
    print(f"Recall: {final_val_metrics['recall']:.4f}")
    print(f"F1 Score: {final_val_metrics['f1']:.4f}")
    print(f"Prediction Time: {final_val_metrics['pred_time']:.4f} seconds")

    print("\nTest Metrics:")
    print(f"Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"Precision: {final_test_metrics['precision']:.4f}")
    print(f"Recall: {final_test_metrics['recall']:.4f}")
    print(f"F1 Score: {final_test_metrics['f1']:.4f}")
    print(f"Prediction Time: {final_test_metrics['pred_time']:.4f} seconds")

    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")
    print("===== Training Completed =====")