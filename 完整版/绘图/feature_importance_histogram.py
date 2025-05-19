import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 定义数据
data = {
    'Sample Size': [200, 600, 1000, 34200],
    'K1K2 Signal': [4.0823, 4.0824, 4.0825, 4.1382],
    'Electronic Lock Signal': [3.9905, 3.9905, 3.9905, 3.9905],
    'Emergency Stop Signal': [3.79, 3.7903, 3.7908, 3.9372],
    'Access Control Signal': [4.2646, 4.2646, 4.2646, 4.2646],
    'THDV-M': [0.2209, 0.221, 0.221, 0.2219],
    'THDI-M': [10.7837, 10.5147, 8.7134, 11.5922]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 设置图形大小
plt.figure(figsize=(768/100, 576/100))

# 定义指标和颜色
metrics = ['K1K2 Signal', 'Electronic Lock Signal', 'Emergency Stop Signal', 'Access Control Signal', 'THDV-M', 'THDI-M']
colors = ['blue', 'orange', 'yellow', 'purple', 'green', 'red']

# 计算柱子的宽度和位置
num_samples = len(df['Sample Size'])
num_metrics = len(metrics)
width = 0.8 / num_metrics
x = np.arange(num_samples)

# 绘制每个指标的柱子
for i, metric in enumerate(metrics):
    plt.bar(x + i * width, df[metric], width, label=metric, color=colors[i])

# 设置图形标签和标题
plt.xlabel('Sample size', fontsize=12)  # 修改横坐标标题字体大小
plt.ylabel('Importance of Features (log10)', fontsize=12)  # 修改纵坐标标题字体大小
# plt.title('Feature Importance by Sample Size', fontsize=12)  # 添加标题并设置字体大小
plt.xticks(x + width * (num_metrics - 1) / 2, df['Sample Size'], fontsize=12)  # 修改横坐标刻度字体大小

# 设置图例横向排列并调整位置，让其在图表内合适位置
plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.0), loc='upper center', prop={'size': 6})  # 修改图例位置和字体大小，允许换行

# 添加保存图片功能
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')  # 保存图片

plt.show()
