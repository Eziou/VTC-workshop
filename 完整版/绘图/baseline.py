import matplotlib.pyplot as plt
import numpy as np

# 数据
sample_sizes = [200, 600, 1000]
slgbm = [0.9955, 0.9960, 0.9962]
maml = [0.895, 0.925, 0.830]
pn = [0.856, 0.885, 0.855]
rn = [0.895, 0.910, 0.865]
svm = [0.777, 0.805, 0.814]
lgbm = [0.833, 0.855, 0.872]

plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, slgbm, 'o-', label='SLGBM', color='blue')
plt.plot(sample_sizes, maml, 's-', label='MAML', color='green')
plt.plot(sample_sizes, pn, '^-', label='PN', color='gray')
plt.plot(sample_sizes, rn, 'v-', label='RN', color='cyan')
plt.plot(sample_sizes, svm, '*-', label='SVM', color='purple')
plt.plot(sample_sizes, lgbm, 'd-', label='LGBM', color='orange')

plt.xlabel('Sample Size', fontsize=12)
plt.ylabel('F1-score', fontsize=12)
plt.title('Performance Comparison in Few-shot Scenarios', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('fig2_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()