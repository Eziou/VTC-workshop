
import os
import pandas as pd
from sklearn.metrics import mean_squared_error

# 定义文件路径
file_path = '/Users/han/Charging station/data base/'

# 获取文件列表
csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]

# 读取所有CSV文件到DataFrame列表中
dataframes = [pd.read_csv(os.path.join(file_path, f)) for f in csv_files]

# 计算两个DataFrame之间的相似度
def calculate_similarity(df1, df2):
    # 使用集合来比较 id 列
    set1 = set(df1['id'])
    set2 = set(df2['id'])
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    id_similarity = len(intersection) / len(union) if union else 0
    return id_similarity

# 计算所有文件之间的相似度
similarities = {}
for i in range(len(dataframes)):
    for j in range(i + 1, len(dataframes)):
        sim = calculate_similarity(dataframes[i], dataframes[j])
        similarities[(csv_files[i], csv_files[j])] = sim

# 输出每个文件之间的相似度百分比
for (file1, file2), sim in similarities.items():
    print(f"相似度 ({file1}, {file2}): {sim * 100:.2f}%")

# 计算总体相似度（平均相似度）
overall_similarity = sum(similarities.values()) / len(similarities)
print(f"总体相似度: {overall_similarity * 100:.2f}%")


