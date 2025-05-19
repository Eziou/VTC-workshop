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

# 初始化保存结果的列表
evaluation_results = []

# 测试不同的训练数据量（从10%到100%）
for fraction in range(1, 11):  # 1 to 10, representing 10%, 20%, ..., 100%
    # 选择训练集的不同比例
    train_fraction_size = fraction / 10.0  # 保证在 [0.0, 1.0] 范围内

    # 避免train_fraction_size为1.0时传入train_size=1.0的情况
    if train_fraction_size == 1.0:
        train_fraction_size = 0.9  # 这样避免train_size=1.0的情况

    # 选择训练集的不同比例
    x_train_fraction, _, y_train_fraction, _ = train_test_split(x_train, y_train, train_size=train_fraction_size,
                                                                random_state=42)

    # 创建新的数据集
    lgb_train_fraction = lgb.Dataset(x_train_fraction, y_train_fraction)
    lgb_eval_fraction = lgb.Dataset(x_test, y_test, reference=lgb_train_fraction)

    # 训练模型
    gbm_fraction = lgb.train(param, lgb_train_fraction, num_boost_round=500, valid_sets=lgb_eval_fraction)

    # 预测
    y_predict_test = gbm_fraction.predict(x_test)
    y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_predict_test]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    # 保存结果
    evaluation_results.append({
        'Data Fraction': f'{fraction * 10}%',
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    })

# 将结果转换为 DataFrame
evaluation_df = pd.DataFrame(evaluation_results)

# 输出表格
print(evaluation_df)

# 将评估结果输出为图表
plt.figure(figsize=(10, 6))
evaluation_df.plot(x='Data Fraction', y=['Accuracy', 'Recall', 'Precision', 'F1 Score'], kind='line', marker='o')
plt.title('Model Performance with Varying Data Fraction')
plt.xlabel('Training Data Fraction')
plt.ylabel('Score')
plt.grid(True)
plt.legend(title='Metrics')
plt.show()
