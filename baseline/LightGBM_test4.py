import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 数据读取与预处理
english_col_names = [
    "ID", "K1K2 Signal", "Electronic Lock Signal",
    "Emergency Stop Signal", "Access Control Signal",
    "THDV-M", "THDI-M", "Label"
]

# 更新数据的列名为英文
data = pd.read_csv("/Users/han/Charging station/data/data.csv", names=english_col_names)
dataset_X = data[["K1K2 Signal", "Electronic Lock Signal",
                  "Emergency Stop Signal", "Access Control Signal",
                  "THDV-M", "THDI-M"]].values
dataset_Y = data[["Label"]].values.flatten()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(dataset_X, dataset_Y, test_size=0.2, random_state=42)

# 设置 LightGBM 超参数
param = {
    'max_depth': 8,
    'num_leaves': 16,
    'learning_rate': 0.4,
    'scale_pos_weight': 1,
    'num_threads': 8,
    'objective': 'binary',
    'bagging_fraction': 1,
    'bagging_freq': 1,
    'min_sum_hessian_in_leaf': 0.01,
    'is_unbalance': 'true',
    'metric': 'auc'
}

# 特征重要性排序（假设的特征重要性排序）
feature_importance = np.array([0.2, 0.15, 0.1, 0.25, 0.2, 0.1])
sorted_feature_indexes = np.argsort(feature_importance)[::-1]  # 按重要性降序排列索引

# 测试不同的训练数据量（从100条到1000条，每100条增加）
data_sizes = range(100, 4001, 100)  # 从100到1000，步长为100

# 初始化结果字典
evaluation_results = {
    'Accuracy': np.zeros((len(data_sizes), 6)),  # len(data_sizes)个数据量和6个特征数
    'Recall': np.zeros((len(data_sizes), 6)),
    'Precision': np.zeros((len(data_sizes), 6)),
    'F1 Score': np.zeros((len(data_sizes), 6))
}

for idx, data_size in enumerate(data_sizes):
    # 选择训练集的不同数量
    x_train_fraction = x_train[:data_size]
    y_train_fraction = y_train[:data_size]

    # 逐步移除特征并训练模型
    for i in range(6, 0, -1):  # 从6个特征开始，到1个特征
        # 根据特征重要性排序选择前i个特征
        selected_features = sorted_feature_indexes[:i]
        x_train_selected = x_train_fraction[:, selected_features]
        x_test_selected = x_test[:, selected_features]

        # 创建新的数据集
        lgb_train_fraction = lgb.Dataset(x_train_selected, y_train_fraction)
        lgb_eval_fraction = lgb.Dataset(x_test_selected, y_test, reference=lgb_train_fraction)

        # 训练模型
        gbm_fraction = lgb.train(param, lgb_train_fraction, num_boost_round=500, valid_sets=lgb_eval_fraction)

        # 预测
        y_predict_test = gbm_fraction.predict(x_test_selected)
        y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_predict_test]

        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)

        # 保存结果
        evaluation_results['Accuracy'][idx, i - 1] = accuracy
        evaluation_results['Recall'][idx, i - 1] = recall
        evaluation_results['Precision'][idx, i - 1] = precision
        evaluation_results['F1 Score'][idx, i - 1] = f1

# 绘制3D图表
fig = plt.figure(figsize=(20, 15))

# 四个评估指标
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
for idx, metric in enumerate(metrics):
    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

    # 创建数据网格
    Y, X = np.meshgrid(range(6, 0, -1), range(1, len(data_sizes) + 1))  # 确保 X 和 Y 与 Z 对应
    Z = evaluation_results[metric]  # 获取对应的评估指标数据

    # 使用 `X`, `Y`, `Z` 绘制3D曲面
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # 设置标题和轴标签
    ax.set_title(f'{metric} with Varying Data Size and Feature Count')
    ax.set_xlabel('Data Size (100s)')
    ax.set_ylabel('Number of Features')
    ax.set_zlabel(metric)

    # 标注数据点的具体数值
    for i in range(len(data_sizes)):
        for j in range(6):
            ax.text(X[i, j], Y[i, j], Z[i, j], f'{Z[i, j]:.2f}', color='black', fontsize=8)

plt.tight_layout()
plt.show()

# 输出评估指标的表格
accuracy_df = pd.DataFrame(evaluation_results['Accuracy'], columns=[6, 5, 4, 3, 2, 1],
                           index=[f'{size}' for size in data_sizes])
recall_df = pd.DataFrame(evaluation_results['Recall'], columns=[6, 5, 4, 3, 2, 1],
                         index=[f'{size}' for size in data_sizes])
precision_df = pd.DataFrame(evaluation_results['Precision'], columns=[6, 5, 4, 3, 2, 1],
                            index=[f'{size}' for size in data_sizes])
f1_score_df = pd.DataFrame(evaluation_results['F1 Score'], columns=[6, 5, 4, 3, 2, 1],
                           index=[f'{size}' for size in data_sizes])

# 打印表格
print("Accuracy Table:")
print(accuracy_df)
print("\nRecall Table:")
print(recall_df)
print("\nPrecision Table:")
print(precision_df)
print("\nF1 Score Table:")
print(f1_score_df)


