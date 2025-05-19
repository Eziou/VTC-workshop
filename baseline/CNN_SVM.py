import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from pyswarm import pso  # PSO 实现
import os
import joblib

# 导入 time 库
import time

# 数据加载函数
def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 数据预处理：去除id和label列
    X_train = df_train.drop(columns=['id', 'label']).values
    y_train = df_train['label'].values
    X_test = df_test.drop(columns=['id', 'label']).values
    y_test = df_test['label'].values
    ids_test = df_test['id'].values

    # 标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, ids_test, df_test

# 数据重塑函数
def reshape_data(X_train, X_test, feature_map_size):
    input_dim = X_train.shape[1]
    padded_input_dim = feature_map_size * feature_map_size

    # 填充特征到最近的完全平方数
    X_train_padded = np.pad(X_train, ((0, 0), (0, padded_input_dim - input_dim)), mode='constant')
    X_test_padded = np.pad(X_test, ((0, 0), (0, padded_input_dim - input_dim)), mode='constant')

    X_train_reshaped = X_train_padded.reshape(-1, 1, feature_map_size, feature_map_size)
    X_test_reshaped = X_test_padded.reshape(-1, 1, feature_map_size, feature_map_size)

    # 转换为torch tensor
    X_train_tensor = torch.tensor(X_train_reshaped, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_reshaped, dtype=torch.float32)

    return X_train_tensor, X_test_tensor

# CNN模型定义
class CNN(nn.Module):
    def __init__(self, input_channels, feature_map_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 计算池化后的尺寸，动态适配
        after_pool1 = feature_map_size // 2
        if after_pool1 <= 0:
            raise ValueError(f"Feature map size too small: {after_pool1} after pooling.")

        self.fc_input_size = 16 * after_pool1 * after_pool1
        self.fc1 = nn.Linear(self.fc_input_size, 128)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

# 提取CNN特征
def extract_features(model, loader):
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for data, target in loader:
            features = model(data)
            features_list.append(features.numpy())
            labels_list.append(target.numpy())
    features = np.vstack(features_list)
    labels = np.hstack(labels_list)
    return features, labels

# PSO 优化函数
def pso_objective(params, train_features, train_labels):
    C, gamma = params
    svm_model = SVC(kernel='rbf', C=C, gamma=gamma)
    svm_model.fit(train_features, train_labels)
    predictions = svm_model.predict(train_features)
    f1 = f1_score(train_labels, predictions)
    return -f1  # PSO 目标是最小化，因此取负值

# 主程序
def main():
    # 路径
    train_path = '/Users/han/Charging station/data/train_data.csv'
    test_path = '/Users/han/Charging station/data/test_data.csv'
    incorrect_save_path = '/Users/han/Charging station/data base/CNN_SVM_incorrect_samples.csv'
    cnn_model_path = '/Users/han/Charging station/baseline-model/CNN_model.pth'
    svm_model_path = '/Users/han/Charging station/baseline-model/SVM_model.pkl'

    # 加载数据
    X_train, y_train, X_test, y_test, ids_test, df_test = load_data(train_path, test_path)

    # 数据重塑
    feature_map_size = int(np.ceil(np.sqrt(X_train.shape[1])))
    X_train_tensor, X_test_tensor = reshape_data(X_train, X_test, feature_map_size)

    # 转换为数据加载器
    batch_size = 64
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化CNN模型
    model = CNN(input_channels=1, feature_map_size=feature_map_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 记录训练开始时间
    start_train_time = time.time()  # 记录训练开始时间
    print('Start training CNN model...')

    # 训练CNN
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            features = model(data)
            loss = criterion(features, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/10, Loss: {running_loss / len(train_loader):.4f}')

    # 记录训练结束时间
    end_train_time = time.time()  # 记录训练结束时间
    train_time = end_train_time - start_train_time  # 计算训练时间

    # 提取训练和测试特征
    train_features, train_labels = extract_features(model, train_loader)
    test_features, test_labels = extract_features(model, test_loader)

    # 使用 PSO 优化 SVM 参数
    lb = [0.01, 0.0001]
    ub = [100, 1]
    best_params, _ = pso(
        pso_objective,
        lb,
        ub,
        args=(train_features, train_labels),
        swarmsize=10,
        maxiter=5
    )
    print(f"Best SVM parameters from PSO: C={best_params[0]:.4f}, gamma={best_params[1]:.4f}")

    # 使用最佳参数训练 SVM
    svm_model = SVC(kernel='rbf', C=best_params[0], gamma=best_params[1], probability=True)
    svm_model.fit(train_features, train_labels)

    # 记录预测开始时间
    start_predict_time = time.time()  # 记录开始预测的时间
    print('Start predicting with SVM model...')

    # 测试 SVM
    y_pred = svm_model.predict(test_features)

    # 记录预测结束时间
    end_predict_time = time.time()  # 记录预测结束的时间
    predict_time = end_predict_time - start_predict_time  # 计算预测时间

    # 计算评估指标
    accuracy = accuracy_score(test_labels, y_pred)
    recall = recall_score(test_labels, y_pred)
    precision = precision_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred)

    # 打印评估结果
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # 打印训练和预测时间
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Prediction time: {predict_time:.4f} seconds")

    # 保存未正确分类的样本
    incorrect_indices = np.where(y_pred != test_labels)[0]
    incorrect_ids = ids_test[incorrect_indices]
    incorrect_samples = df_test[df_test['id'].isin(incorrect_ids)]
    incorrect_samples.to_csv(incorrect_save_path, index=False)
    print(f"Incorrect samples saved to {incorrect_save_path}")

    # 保存CNN模型
    torch.save(model.state_dict(), cnn_model_path)
    print(f"CNN model saved to {cnn_model_path}")

    # 保存SVM模型
    joblib.dump(svm_model, svm_model_path)
    print(f"SVM model saved to {svm_model_path}")

if __name__ == '__main__':
    main()
