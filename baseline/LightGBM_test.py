import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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

# 创建 LightGBM 数据集
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

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

# 训练模型
print("Start training...")
gbm = lgb.train(param, lgb_train, num_boost_round=500, valid_sets=lgb_eval)

# 获取特征重要性
feature_importance = gbm.feature_importance(importance_type='split')
feature_names = ["K1K2 Signal", "Electronic Lock Signal",
                 "Emergency Stop Signal", "Access Control Signal",
                 "THDV-M", "THDI-M"]

# 将特征重要性和特征名称组合成一个DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# 根据特征重要性从低到高排序
importance_df = importance_df.sort_values(by='Importance', ascending=True)

# 用于存储每次特征选择后的评估结果
evaluation_results = []

# 从特征重要性最低的开始逐步移除特征
for i in range(len(feature_names)):
    # 获取剩余的特征
    selected_features = importance_df['Feature'].iloc[i:].values
    x_train_selected = x_train[:, [feature_names.index(f) for f in selected_features]]
    x_test_selected = x_test[:, [feature_names.index(f) for f in selected_features]]

    # 创建新的数据集
    lgb_train_selected = lgb.Dataset(x_train_selected, y_train)
    lgb_eval_selected = lgb.Dataset(x_test_selected, y_test, reference=lgb_train_selected)

    # 训练新模型
    gbm_selected = lgb.train(param, lgb_train_selected, num_boost_round=500, valid_sets=lgb_eval_selected)

    # 预测
    y_predict_test = gbm_selected.predict(x_test_selected)
    y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_predict_test]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    # 保存结果
    evaluation_results.append({
        'Number of Features': len(selected_features),
        'Selected Features': ', '.join(selected_features),
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    })

# 输出评估结果表格
evaluation_df = pd.DataFrame(evaluation_results)

# 输出为表格
print(evaluation_df)

# 创建四个子图，每个评估指标一个子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 评估指标名称
metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
markers = ['o', '^', 's', '*']  # 不同的图标样式

# 绘制每个评估指标的图
for idx, metric in enumerate(metrics):
    ax = axs[idx // 2, idx % 2]  # 使用二维索引分配子图
    ax.plot(evaluation_df['Number of Features'], evaluation_df[metric], marker=markers[idx], label=metric)
    ax.set_title(f'{metric} vs. Number of Features')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel(f'{metric}')
    ax.grid(True)
    ax.legend()

# 调整布局，使得子图不重叠
plt.tight_layout()
plt.show()
