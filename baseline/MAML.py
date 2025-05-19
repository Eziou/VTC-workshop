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

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, X.shape[1]


# 分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # 二分类输出
        )

    def forward(self, x):
        return self.network(x)


# 直接训练类
class DirectTrainer:
    def __init__(self, input_dim, lr=0.001):
        self.model = Classifier(input_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_loader, num_epochs=50):
        self.model.train()
        for _ in tqdm(range(num_epochs), desc="Direct Training"):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        start_time = time.time()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        total_time = time.time() - start_time
        per_sample_time = (total_time / len(all_preds)) * 1e6

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'total_time': total_time,
            'per_sample_time_us': per_sample_time
        }
        return metrics


# 主程序
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    X, y, input_dim = load_data()
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

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
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        trainer = DirectTrainer(input_dim=input_dim, lr=0.001)

        start_time = time.time()
        trainer.train(train_loader)
        fold_train_time = time.time() - start_time
        total_train_time += fold_train_time

        if fold == k_folds - 1:
            final_train_metrics = trainer.evaluate(train_loader)
            final_val_metrics = trainer.evaluate(val_loader)
            final_test_metrics = trainer.evaluate(test_loader)

    print("\n===== Final Results =====")
    print("Train Metrics:")
    print(f"Accuracy: {final_train_metrics['accuracy']:.4f}")
    print(f"Precision: {final_train_metrics['precision']:.4f}")
    print(f"Recall: {final_train_metrics['recall']:.4f}")
    print(f"F1 Score: {final_train_metrics['f1']:.4f}")
    print(
        f"Prediction Time - Total: {final_train_metrics['total_time']:.4f}s, Per Sample: {final_train_metrics['per_sample_time_us']:.2f}us")

    print("\nValidation Metrics:")
    print(f"Accuracy: {final_val_metrics['accuracy']:.4f}")
    print(f"Precision: {final_val_metrics['precision']:.4f}")
    print(f"Recall: {final_val_metrics['recall']:.4f}")
    print(f"F1 Score: {final_val_metrics['f1']:.4f}")
    print(
        f"Prediction Time - Total: {final_val_metrics['total_time']:.4f}s, Per Sample: {final_val_metrics['per_sample_time_us']:.2f}us")

    print("\nTest Metrics:")
    print(f"Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"Precision: {final_test_metrics['precision']:.4f}")
    print(f"Recall: {final_test_metrics['recall']:.4f}")
    print(f"F1 Score: {final_test_metrics['f1']:.4f}")
    print(
        f"Prediction Time - Total: {final_test_metrics['total_time']:.4f}s, Per Sample: {final_test_metrics['per_sample_time_us']:.2f}us")

    print(f"\nTotal Training Time: {total_train_time:.2f} seconds")
    print("===== Training Completed =====")