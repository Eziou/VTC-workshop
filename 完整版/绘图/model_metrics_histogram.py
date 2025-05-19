import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 三组数据
data_list = [
    {
        'Models': ['MAML', 'PN', 'RN', 'SVM', 'LGBM', 'SLGBM'],
        'Accuracy': [0.9, 0.85, 0.9, 0.8, 0.85, 0.995544],
        'Recall': [0.85, 0.9, 0.85, 0.7, 0.75, 0.9967],
        'Precision': [0.9444, 0.8182, 0.9494, 0.875, 0.9375, 0.994566],
        'F1-score': [0.8947, 0.8571, 0.8947, 0.7778, 0.8333, 0.9955],
        'Time(s)': [0.0015, 0.0124, 0.0074, 0.0005, 0.0049, 0.0244]
    },
    {
        'Models': ['MAML', 'PN', 'RN', 'SVM', 'LGBM', 'SLGBM'],
        'Accuracy': [0.825, 0.8833, 0.85, 0.7667, 0.9, 0.995799],
        'Recall': [0.85, 0.8833, 0.8833, 0.7833, 0.9167, 0.9969],
        'Precision': [0.8095, 0.8833, 0.8621, 0.7581, 0.8871, 0.9944],
        'F1-score': [0.8293, 0.8833, 0.8475, 0.7705, 0.9016, 0.9958],
        'Time(s)': [0.0032, 0.0025, 0.0035, 0.0019, 0.0041, 0.0245]
    },
    {
        'Models': ['MAML', 'PN', 'RN', 'SVM', 'LGBM', 'SLGBM'],
        'Accuracy': [0.83, 0.86, 0.845, 0.805, 0.875, 0.9982],
        'Recall': [0.83, 0.83, 0.83, 0.85, 0.86, 0.9976],
        'Precision': [0.83, 0.8829, 0.8557, 0.7798, 0.8866, 0.9948],
        'F1-score': [0.83, 0.8557, 0.8426, 0.8134, 0.8731, 0.9962],
        'Time(s)': [0.0029, 0.0046, 0.0033, 0.0036, 0.0063, 0.0252]
    }
]

# 调整为指定的色系
color_mapping = {
    'Accuracy': 'blue',
    'Recall': 'orange',
    'Precision': 'yellow',
    'F1-score': 'purple',
    'Time(s)': 'green'
}

# 样本量列表
sample_sizes = [200, 600, 1000]

for i, data in enumerate(data_list):
    df = pd.DataFrame(data)
    fig, ax1 = plt.subplots(figsize=(768/100, 576/100))

    # 统一柱子宽度
    bar_width = 0.15
    x = np.arange(len(df['Models']))

    # 绘制Accuracy、Recall、Precision、F1-score指标，在左纵坐标
    metrics_left = ['Accuracy', 'Recall', 'Precision', 'F1-score']
    for j, metric in enumerate(metrics_left):
        ax1.bar(x + j * bar_width, df[metric], width=bar_width, label=metric, color=color_mapping[metric])
    ax1.set_xlabel('Models', fontsize=12)  # 修改横坐标标题字体大小
    ax1.set_ylabel('Evaluation Metrics Values', color='black', fontsize=12)  # 修改纵坐标标题字体大小
    ax1.set_xticks(x + (len(metrics_left) - 1) * bar_width / 2)
    ax1.set_xticklabels(df['Models'], fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')

    # 绘制Time指标，在右纵坐标
    ax2 = ax1.twinx()
    ax2.bar(x + len(metrics_left) * bar_width, df['Time(s)'], width=bar_width, label='Time(s)', color=color_mapping['Time(s)'])
    ax2.set_ylabel('Time (s)', color='black', fontsize=12)  # 修改纵坐标标题字体大小
    ax2.tick_params(axis='y', labelcolor='black')

    # 添加图例，设置为横向排列，并放到图框正上方
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=5,bbox_to_anchor=(0.5, 1.0), loc='upper center', prop={'size': 6})  # 修改图例位置和字体大小

    # # 设置标题
    # plt.title(f'Model Metrics by Sample Size {sample_sizes[i]}', fontsize=12)

    # 添加保存图片功能
    plt.savefig(f'fewshot_{sample_sizes[i]}.png', dpi=300, bbox_inches='tight')  # 保存图片

    plt.show()
