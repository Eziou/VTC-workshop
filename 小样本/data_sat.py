import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 路径配置
DATA_PATH = '/home/user/hrx/CSdata/data.csv'
SAVE_DIR = '/home/user/hrx/小样本数据集'
os.makedirs(SAVE_DIR, exist_ok=True)

# 列名
COLUMN_NAMES = [
    "ID", "K1K2 Signal", "Electronic Lock Signal",
    "Emergency Stop Signal", "Access Control Signal",
    "THDV-M", "THDI-M", "Label"
]

# 读取原始数据集
df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)

# 分离正常和异常样本
normal_samples = df[df['Label'] == 0]  # 正常样本 (Label=0)
abnormal_samples = df[df['Label'] == 1]  # 异常样本 (Label=1)

# 样本量和比例
sample_sizes = [200, 400, 600, 800, 1000]
ratios = {
    '1_1': (1, 1),  # 正常:异常 = 1:1
    '1_4': (4, 1),  # 正常:异常 = 4:1
    '1_9': (9, 1)   # 正常:异常 = 9:1
}

# 筛选和划分数据集
for size in sample_sizes:
    for ratio_name, (normal_ratio, abnormal_ratio) in ratios.items():
        # 计算正常和异常样本数量
        total_ratio = normal_ratio + abnormal_ratio
        normal_count = int(size * normal_ratio / total_ratio)
        abnormal_count = size - normal_count

        # 随机抽样
        normal_subset = normal_samples.sample(n=normal_count, random_state=42)
        abnormal_subset = abnormal_samples.sample(n=abnormal_count, random_state=42)

        # 合并数据集
        subset = pd.concat([normal_subset, abnormal_subset], axis=0).sample(frac=1, random_state=42)

        # 划分训练集和测试集 (80%训练，20%测试)
        train, test = train_test_split(subset, test_size=0.2, stratify=subset['Label'], random_state=42)

        # 保存
        train_path = os.path.join(SAVE_DIR, f'ratio_{ratio_name}_train_{size}.csv')
        test_path = os.path.join(SAVE_DIR, f'ratio_{ratio_name}_test_{size}.csv')
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        print(f'Saved: {train_path}, {test_path}')