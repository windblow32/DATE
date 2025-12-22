import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 24})
# 设置Arial字体
plt.rcParams['font.family'] = 'Arial'


# 数据
datasets = ['BM', 'EL', 'MT', 'EM', 'HE', 'CR', 'CA', 'JA']
real_data = [54.44, 60.00, 48.65, 64.28, 56.25, 63.74, 71.43, 54.55]
date_data = [41.13, 54.29, 39.47, 58.83, 56.25, 42.86, 57.14, 36.36]

x = np.arange(len(datasets))  # 标签位置
width = 0.4 # 柱子的宽度

# 增加上边距，为图例留出空间
fig, ax = plt.subplots(figsize=(14, 7))
# plt.subplots_adjust(top=0.85)  # 调整上边距

# 绘制柱状图
rects1 = ax.bar(x - width/2, real_data, width, label='Real Data', color='#FAE3A8', alpha=0.7, hatch='//', edgecolor='#050301', linewidth=0.5)
rects2 = ax.bar(x + width/2, date_data, width, label='DATE', color='#FEB86B', alpha=0.7, hatch='\\\\', edgecolor='#050301', linewidth=0.5)

# 添加文本、标签等
ax.set_xlabel('Dataset', fontsize=24)
ax.set_ylabel('Error Rate (%)', fontsize=24)
ax.set_xticks(x)
ax.set_xticklabels(datasets, ha='center', fontsize=24)
ax.tick_params(axis='y', labelsize=24)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# 在柱子顶部显示数值
# ax.bar_label(rects1, padding=3, fmt='%.1f%%', fontsize=12)
# ax.bar_label(rects2, padding=3, fmt='%.1f%%', fontsize=12)

# 将图例放在图外顶部居中
legend = ax.legend(bbox_to_anchor=(0.5, 1),  # 调整y轴偏移量
                  loc='lower center',
                  ncol=2,  # 水平排列图例
                  borderaxespad=0.,
                  frameon=False,
                  fontsize=24)  # 图例字体大小

# 设置y轴范围
ax.set_ylim(0, max(real_data + date_data) * 1.2)

fig.tight_layout()
plt.savefig('dpo.eps')
plt.show()
