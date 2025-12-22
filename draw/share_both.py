import matplotlib.pyplot as plt
import numpy as np

# 设置Arial字体
plt.rcParams['font.family'] = 'Arial'

# 创建包含两个子图的图形
# 创建包含两个子图的图形，增加水平间距
plt.rcParams['figure.constrained_layout.use'] = True
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10), gridspec_kw={'wspace': 0.3})

# ===== 第一个子图：Time and Error =====
# 数据准备
datasets1 = ['MT', 'CA', 'EL', 'BO', 'EM', 'CR', 'HE', 'JA', 'EM', 'BI']

# 无加速策略数据
time_no_accel = [1719.13, 1883.79, 7213.52, 2809, 31.79, 1649.14, 301.78, 7205.69, 7211.41, 34]
error_no_accel = [17.69, 13.48, 10.51, 19, 22.61, 27.24, 31.98, 30.45, 41.80, 7.54]

# 加速策略数据  
time_accel = [420, 516, 1980, 730.5, 27, 448, 38, 1740, 670, 30]
error_accel = [17.15, 13.72, 13.10, 20.79, 22.98, 25.03, 34.72, 27.75, 44.03, 7.54]

# 设置位置和宽度
x1 = np.arange(len(datasets1))
width = 0.5  # 增加柱状图宽度

# 在左侧y轴绘制柱状图（时间）
bars1 = ax1.bar(x1 - width/2, time_no_accel, width, label='w/o model sharing: time', 
               color='#C94E65', alpha=0.7, hatch='//', edgecolor='#050301', linewidth=0.5)
bars2 = ax1.bar(x1 + width/2, time_accel, width, label='model sharing: time', 
               color='#EAB080', alpha=0.7, hatch='\\\\', edgecolor='#050301', linewidth=0.5)

ax1.set_xlabel('Datasets', fontsize=24)
ax1.set_ylabel('Time (s)', fontsize=24, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=22)
ax1.set_xticks(x1)
ax1.set_xticklabels(datasets1, rotation=0, ha='center', fontsize=23)
ax1.set_yscale('log')  # 使用对数坐标因为时间跨度很大
# ax1.set_title('(a) Time and Error Comparison', fontsize=24, pad=20)

# 创建第二个y轴用于误差
ax1b = ax1.twinx()

# 绘制折线图（误差）
line1 = ax1b.plot(x1, error_no_accel, color='#C94E64', marker='*', markersize=15, 
                 linewidth=5, label='w/o model sharing: error', 
                 markerfacecolor='white', markeredgewidth=2, zorder=5)
line2 = ax1b.plot(x1, error_accel, color='#E79D37', marker='o', markersize=15, 
                 linewidth=5, label='model sharing: error', 
                 markerfacecolor='none', markeredgewidth=2, zorder=5)

ax1b.set_ylabel('Error (%)', fontsize=24, color='black')
ax1b.tick_params(axis='y', labelcolor='black', labelsize=22)

# 组合图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='upper center', bbox_to_anchor=(0.5, 1.2),  # 调整图例位置
          ncol=2, fancybox=True, shadow=False, frameon=False, fontsize=22, columnspacing=0.8)

# ===== 第二个子图：Subset and Tokens =====
# 数据准备
datasets2 = ['MT', 'CA', 'EL', 'BO', 'EM', 'CR', 'HE', 'JA', 'EM', 'BI']

# 子集数和令牌数数据
subset_no_accel = [46, 43, 23, 23, 6, 14, 13, 8, 19, 2]
tokens_no_accel = [82702, 75042, 38820, 42912, 9937, 27840, 34344, 26157, 42690, 3972]

subset_accel = [3, 2, 2, 3, 4, 3, 2, 3, 3, 2]
tokens_accel = [5474, 3579, 3435, 5652, 6635, 6170, 5733, 9942, 6930, 3972]

# 设置位置和宽度
x2 = np.arange(len(datasets2))
width = 0.5  # 增加柱状图宽度

# 在左侧y轴绘制柱状图（子集数量）
bars3 = ax2.bar(x2 - width/2, subset_no_accel, width, 
               label='w/o model sharing: subsets', 
               color='#4E79A7', alpha=0.7, hatch='//', 
               edgecolor='#050301', linewidth=0.5)
bars4 = ax2.bar(x2 + width/2, subset_accel, width, 
               label='model sharing: subsets', 
               color='#F28E2B', alpha=0.7, hatch='\\\\', 
               edgecolor='#050301', linewidth=0.5)

ax2.set_xlabel('Datasets', fontsize=24)
ax2.set_ylabel('Subset Number', fontsize=24, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=22)
ax2.set_xticks(x2)
ax2.set_xticklabels(datasets2, rotation=0, ha='center', fontsize=22)
# ax2.set_title('(b) Subset and Tokens Comparison', fontsize=24, pad=20)

# 创建第二个y轴用于令牌数
ax2b = ax2.twinx()

# 绘制折线图（令牌数）
line3 = ax2b.plot(x2, tokens_no_accel, color='#4E79A7', marker='s', markersize=15, 
                 linewidth=5, label='w/o model sharing: tokens', 
                 markerfacecolor='white', markeredgewidth=2, zorder=5)
line4 = ax2b.plot(x2, tokens_accel, color='#F28E2B', marker='^', markersize=15, 
                 linewidth=5, label='model sharing: tokens', 
                 markerfacecolor='none', markeredgewidth=2, zorder=5)

ax2b.set_ylabel('Number of Tokens', fontsize=24, color='black')
ax2b.tick_params(axis='y', labelcolor='black', labelsize=22)

# 将y轴刻度转换为k格式
def format_k(x, pos):
    return f'{x/1000:.0f}k'

ax2b.yaxis.set_major_formatter(plt.FuncFormatter(format_k))

# 组合图例
lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2b.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, 
          loc='upper center', bbox_to_anchor=(0.5, 1.2),  # 调整图例位置
          ncol=2, fancybox=True, shadow=False, frameon=False, fontsize=22, columnspacing=0.8)

# 调整布局并保存
plt.subplots_adjust(wspace=0.5)  # 进一步增加子图之间的水平间距
plt.tight_layout(rect=[0.056, 0.079, 0.944, 0.91])  # 调整整体布局，为图例留出更多空间

# 保存为EPS和PNG
output_eps = 'share_both.eps'
output_png = 'share_both.png'
plt.savefig(output_eps, format='eps', dpi=300, bbox_inches='tight')
plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')

print(f"Combined figure saved as {output_eps} and {output_png}")

# 显示图形
plt.show()