import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 配置路径
DATA_DIR = '/home/user/hrx/完整版/预训练/data'
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
PRETRAIN_DATA_PATH = os.path.join(DATA_DIR, 'pretrain_data.csv')
TRAINVAL_DATA_PATH = os.path.join(DATA_DIR, 'trainval_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.csv')

# 读取数据，不指定列名
df = pd.read_csv(ORIGINAL_DATA_PATH, header=None)
print(f"Original data shape: {df.shape}, Label distribution: {df.iloc[:, -1].value_counts().to_dict()}")

# 分割正负样本（最后一列是标签）
df_positive = df[df.iloc[:, -1] == 1]
df_negative = df[df.iloc[:, -1] == 0]
print(f"Positive samples: {df_positive.shape[0]}, Negative samples: {df_negative.shape[0]}")

# 对正负样本分别划分
# 测试集 20%
pos_temp, pos_test = train_test_split(df_positive, test_size=0.2, random_state=42)
neg_temp, neg_test = train_test_split(df_negative, test_size=0.2, random_state=42)

# 剩余数据中，预训练集 37.5%，训练/验证集 62.5%
pos_pretrain, pos_trainval = train_test_split(pos_temp, test_size=0.625, random_state=42)
neg_pretrain, neg_trainval = train_test_split(neg_temp, test_size=0.625, random_state=42)

# 合并正负样本
pretrain_data = pd.concat([pos_pretrain, neg_pretrain])
trainval_data = pd.concat([pos_trainval, neg_trainval])
test_data = pd.concat([pos_test, neg_test])

# 打乱顺序
pretrain_data = pretrain_data.sample(frac=1, random_state=42).reset_index(drop=True)
trainval_data = trainval_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 检查划分结果
print(f"Pretrain data shape: {pretrain_data.shape}, Label distribution: {pretrain_data.iloc[:, -1].value_counts().to_dict()}")
print(f"Trainval data shape: {trainval_data.shape}, Label distribution: {trainval_data.iloc[:, -1].value_counts().to_dict()}")
print(f"Test data shape: {test_data.shape}, Label distribution: {test_data.iloc[:, -1].value_counts().to_dict()}")

# 检查是否有重复数据
assert pretrain_data.duplicated().sum() == 0, "Duplicates in pretrain_data"
assert trainval_data.duplicated().sum() == 0, "Duplicates in trainval_data"
assert test_data.duplicated().sum() == 0, "Duplicates in test_data"
assert len(pretrain_data) + len(trainval_data) + len(test_data) == len(df), "Data split size mismatch"
concat_data = pd.concat([pretrain_data, trainval_data, test_data])
assert concat_data.duplicated().sum() == 0, "Duplicate data found across splits"

# 保存划分后的数据集
pretrain_data.to_csv(PRETRAIN_DATA_PATH, index=False, header=False)
trainval_data.to_csv(TRAINVAL_DATA_PATH, index=False, header=False)
test_data.to_csv(TEST_DATA_PATH, index=False, header=False)
print(f"Data saved to {DATA_DIR}: pretrain_data.csv, trainval_data.csv, test_data.csv")