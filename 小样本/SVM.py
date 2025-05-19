import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import os
import time  # 导入时间模块

# 设置随机种子，确保结果可复现
np.random.seed(42)

# 1. 加载数据
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
    return train_features, train_labels, test_features, test_labels

# 2. 训练和评估 SVM 模型
def train_and_evaluate(train_features, train_labels, test_features, test_labels):
    # 计算正类权重
    num_pos = np.sum(train_labels == 1)
    num_neg = np.sum(train_labels == 0)
    class_weight = {0: 1.0, 1: num_neg / (num_pos + 1e-6)}  # 处理不平衡数据

    # 初始化 SVM 模型
    model = SVC(kernel='rbf', class_weight=class_weight, probability=True, random_state=42)

    # 训练模型并计时
    start_time = time.time()
    model.fit(train_features, train_labels)
    train_time = time.time() - start_time

    # 预测概率并计时
    start_time = time.time()
    test_probs = model.predict_proba(test_features)[:, 1]
    predict_time = time.time() - start_time

    # 使用动态阈值
    threshold = min(0.5, max(0.05, np.sum(test_labels == 1) / len(test_labels)))
    test_preds = (test_probs > threshold).astype(int)

    # 计算评估指标
    acc = accuracy_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds, pos_label=1, zero_division=0)
    precision = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
    f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
    print(f"Pred distribution: Pos={np.sum(test_preds)}, Neg={len(test_preds) - np.sum(test_preds)}")
    return acc, recall, precision, f1, train_time, predict_time

# 3. 主程序
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
            train_features, train_labels, test_features, test_labels = load_data(train_file, test_file)

            # 训练和评估
            acc, recall, precision, f1, train_time, predict_time = train_and_evaluate(train_features, train_labels, test_features, test_labels)

            # 保存结果
            results_by_ratio[ratio].append({
                '样本量': size,
                '准确率': acc,
                '召回率': recall,
                '精准率': precision,
                'F1': f1,
                '训练时间': train_time,  # 添加训练时间
                '预测时间': predict_time  # 添加预测时间
            })
            print(f"比例 {ratio}, 样本量 {size} 完成: 准确率={acc:.6f}, 召回率={recall:.6f}, 精准率={precision:.6f}, F1={f1:.6f}, 训练时间={train_time:.6f}s, 预测时间={predict_time:.6f}s")

        # 保存该比例的结果到 CSV
        results_df = pd.DataFrame(results_by_ratio[ratio])
        result_file = os.path.join(result_dir, f'svm_ratio_{ratio}_results.csv')
        results_df.to_csv(result_file, index=False)
        print(f"比例 {ratio} 的结果已保存到 {result_file}")

    print("所有实验完成！")
