import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import time  # 导入 time 库，用于计算时间

# 导入数据
col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
data = pd.read_csv("/Users/han/Charging station/data/data.csv", names=col_names)
dataset_X = data[["K1K2驱动信号", "电子锁驱动信号", "急停信号", "门禁信号", "THDV-M", "THDI-M"]].values
dataset_Y = data[["label"]].values.flatten()

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
    'min_sum_hessian_in_leaf': 0.01
}
param['is_unbalance'] = 'true'
param['metric'] = 'auc'

# 记录训练时间
start_train_time = time.time()  # 记录训练开始时间
print('Start training...')

gbm = lgb.train(param,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                )

end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时间

# 记录预测时间
start_predict_time = time.time()  # 记录预测开始时间
print('Start predicting...')

y_predict_test = gbm.predict(x_test)
y_pred_binary = [1 if prob > 0.5 else 0 for prob in y_predict_test]

end_predict_time = time.time()  # 记录预测结束时间
predict_time = end_predict_time - start_predict_time  # 计算预测时间

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

# 打印评估指标
print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Precision: {precision}')
print(f'F1 Score: {f1}')

# 打印训练和预测时间
print(f"Training time: {train_time:.4f} seconds")
print(f"Prediction time: {predict_time:.4f} seconds")
