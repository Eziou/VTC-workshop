import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os
import time  # 导入 time 库

# 加载数据
df_train = pd.read_csv('/Users/han/Charging station/data/train_data.csv')
df_test = pd.read_csv('/Users/han/Charging station/data/test_data.csv')

X_train = df_train.drop(columns=['id', 'label']).values
y_train = df_train['label'].values
X_test = df_test.drop(columns=['id', 'label']).values
y_test = df_test['label'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)


# 定义BP神经网络模型
class BPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)  # 输出 logits
        return out


# 模型评估函数并保存未正确分类样本
def evaluate_and_save_incorrect(model, test_loader, df_test, save_path):
    model.eval()
    y_pred_list = []
    y_true_list = []
    incorrect_indices = []

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(target.cpu().numpy())

            # 获取未正确分类的索引
            batch_indices = torch.arange(idx * data.size(0), (idx + 1) * data.size(0))
            incorrect_batch_indices = batch_indices[(predicted != target).cpu().numpy()]
            incorrect_indices.extend(incorrect_batch_indices.tolist())

    # 计算指标
    accuracy = accuracy_score(y_true_list, y_pred_list)
    recall = recall_score(y_true_list, y_pred_list)
    precision = precision_score(y_true_list, y_pred_list)
    f1 = f1_score(y_true_list, y_pred_list)

    # 保存未正确分类的样本到 CSV
    incorrect_samples = df_test.iloc[incorrect_indices]
    os.makedirs(save_path, exist_ok=True)
    incorrect_samples.to_csv(os.path.join(save_path, 'BP_unrecognized_anomalies.csv'), index=False)
    print(f"Incorrectly classified samples saved to {save_path}/BP_unrecognized_anomalies.csv")

    return accuracy, recall, precision, f1


# 超参数搜索
def find_best_hyperparameters():
    best_f1 = 0
    best_params = {}

    # 搜索空间
    hidden_dims = [64, 128, 256]
    learning_rates = [0.1, 0.01, 0.001]

    for hidden_dim in hidden_dims:
        for lr in learning_rates:
            print(f"Training with hidden_dim={hidden_dim}, learning_rate={lr}")

            # 初始化模型、优化器和损失函数
            input_dim = X_train.shape[1]
            output_dim = 2
            model = BPNN(input_dim, hidden_dim, output_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            # 数据加载器
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

            # 学习率调度器
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5,
                                                             verbose=False)

            # 训练模型
            model.train()
            start_train_time = time.time()  # 记录开始训练的时间
            for epoch in range(20):  # 固定训练轮次
                running_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                scheduler.step(avg_loss)

            end_train_time = time.time()  # 记录训练结束的时间
            train_time = end_train_time - start_train_time  # 计算训练时间

            # 验证模型
            start_predict_time = time.time()  # 记录开始预测的时间
            accuracy, recall, precision, f1 = evaluate_and_save_incorrect(
                model, test_loader, df_test, '/Users/han/Charging station/data base'
            )
            end_predict_time = time.time()  # 记录预测结束的时间
            predict_time = end_predict_time - start_predict_time  # 计算预测时间

            print(f"Hidden Dim: {hidden_dim}, LR: {lr}, F1 Score: {f1:.4f}")
            print(f"Training time: {train_time:.4f} seconds")
            print(f"Prediction time: {predict_time:.4f} seconds")

            # 更新最佳参数
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'hidden_dim': hidden_dim, 'learning_rate': lr}

    print(f"Best Parameters: {best_params}, Best F1: {best_f1:.4f}")
    return best_params

# 主函数
def main():
    # 搜索最佳超参数
    best_params = find_best_hyperparameters()

    # 使用最佳参数重新训练
    hidden_dim = best_params['hidden_dim']
    learning_rate = best_params['learning_rate']

    input_dim = X_train.shape[1]
    output_dim = 2
    model = BPNN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

    # 训练最终模型
    model.train()
    start_train_time = time.time()  # 记录开始训练的时间
    num_epochs = 64
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step(avg_loss)

        # 打印每轮的指标
        start_predict_time = time.time()  # 记录开始预测的时间
        accuracy, recall, precision, f1 = evaluate_and_save_incorrect(
            model, test_loader, df_test, '/Users/han/Charging station/data base'
        )
        end_predict_time = time.time()  # 记录预测结束的时间
        predict_time = end_predict_time - start_predict_time  # 计算预测时间

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, "
              f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
        end_train_time = time.time()  # 记录训练结束的时间
        train_time = end_train_time - start_train_time  # 计算训练时间
        print(f"Training time: {train_time:.4f} seconds")
        print(f"Prediction time: {predict_time:.4f} seconds")

    end_train_time = time.time()  # 记录训练结束的时间
    train_time = end_train_time - start_train_time  # 计算训练时间

    # 保存最终模型
    save_path = '/Users/han/Charging station/baseline-model'
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'BPNN_model.pth'))
    print('Final model saved successfully.')
    print(f"Total Training time: {train_time:.4f} seconds")

if __name__ == '__main__':
    main()