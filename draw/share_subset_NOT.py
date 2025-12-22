import matplotlib.pyplot as plt
import numpy as np

# 设置Arial字体
plt.rcParams['font.family'] = 'Arial'

# 数据准备
datasets = ['EM', 'EL', 'MT', 'EM', 
           'HE', 'CR', 'CA', 'JA', 'BI', 'BO']

# 子集数和令牌数数据
subset_no_accel = [6, 23, 46, 19, 13, 14, 43, 8, 2, 23]
tokens_no_accel = [9937, 38820, 82702, 42690, 34344, 27840, 75042, 26157, 3972, 42912]

subset_accel = [4, 2, 3, 3, 2, 3, 2, 3, 2, 3]  
tokens_accel = [6635, 3435, 5474, 6930, 5733, 6170, 3579, 9942, 3972, 5652]

# 按照无加速策略子集数从大到小排序
sorted_indices = sorted(range(len(subset_no_accel)), key=lambda i: subset_no_accel[i], reverse=True)

# 重新排列数据
datasets = [datasets[i] for i in sorted_indices]
subset_no_accel = [subset_no_accel[i] for i in sorted_indices]
subset_accel = [subset_accel[i] for i in sorted_indices]
tokens_no_accel = [tokens_no_accel[i] for i in sorted_indices]
tokens_accel = [tokens_accel[i] for i in sorted_indices]

# 计算平均改进
avg_subset_no_accel = np.mean(subset_no_accel)
avg_subset_accel = np.mean(subset_accel)
subset_reduction = (avg_subset_no_accel - avg_subset_accel) / avg_subset_no_accel * 100
subset_ratio = avg_subset_no_accel / avg_subset_accel

avg_tokens_no_accel = np.mean(tokens_no_accel)
avg_tokens_accel = np.mean(tokens_accel)
tokens_reduction = (avg_tokens_no_accel - avg_tokens_accel) / avg_tokens_no_accel * 100

print(f'平均子集数减少: {subset_reduction:.1f}% (压缩比: {subset_ratio:.2f}x)')
print(f'平均令牌数减少: {tokens_reduction:.1f}%')

# 创建图形和双纵坐标轴
fig, ax1 = plt.subplots(figsize=(14, 8))

# 设置位置和宽度
x = np.arange(len(datasets))
width = 0.35

# 在左侧y轴绘制柱状图（子集数量）
bars1 = ax1.bar(x - width/2, subset_no_accel, width, label='w/o model sharing: subsets', color='#C94E65', alpha=0.7, hatch='//', edgecolor='#050301', linewidth=0.5)
bars2 = ax1.bar(x + width/2, subset_accel, width, label='model sharing: subsets', color='#EAB080', alpha=0.7, hatch='\\\\', edgecolor='#050301', linewidth=0.5)

ax1.set_xlabel('Datasets', fontsize=24)
ax1.set_ylabel('Subset Number', fontsize=24, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=24)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=0, ha='right', fontsize=24)

# 创建第二个y轴用于令牌数
ax2 = ax1.twinx()

# 绘制折线图（令牌数）
line1 = ax2.plot(x, tokens_no_accel, color='#C94E64', marker='*', markersize=20, linewidth=4, label='w/o model sharing: tokens', markerfacecolor='white', markeredgewidth=2)
line2 = ax2.plot(x, tokens_accel, color='#E79D37', marker='o', markersize=20, linewidth=4, label='model sharing: tokens', markerfacecolor='none', markeredgewidth=2)

ax2.set_ylabel('Number of Tokens', fontsize=24, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=24)

# 将y轴刻度转换为k格式
def format_k(x, pos):
    return f'{x/1000:.0f}k'

ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_k))

# 组合图例并放在图外正上方居中
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='lower center', bbox_to_anchor=(0.5, 1),
          ncol=2, fancybox=True, shadow=False, frameon=False, fontsize=24)

# # 添加改进信息到图表上
# improvement_text = f'Average Subset Reduction: {subset_reduction:.1f}% ({subset_ratio:.1f}x compression)\nAverage Token Reduction: {tokens_reduction:.1f}%'
# ax1.text(0.5, 0.95, improvement_text, transform=ax1.transAxes, 
#          fontsize=18, fontweight='bold', ha='center', va='top',
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig('share_subset_not.eps')
plt.show()