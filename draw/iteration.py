import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
datasets = ['BM', 'EL', 'MT', 'EM', 'HE', 'CR', 'CA', 'JA', 'BI', 'BO']

# 误差数据
error_data = {
    'Iteration 0': [25.83, 13.89, 17.98, 44.37, 36.71, 29.49, 14.38, 28.44, 7.60, 20.79],
    'Iteration 1': [23.00, 12.95, 16.96, 43.85, 34.71, 24.90, 13.59, 27.49, 7.54, 17.86],
    'Iteration 2': [22.31, 12.85, 16.65, 43.65, 34.59, 24.83, 13.20, 27.19, 7.53, 17.86],
    'Iteration 3': [20.86, 12.78, 16.58, 43.10, 34.50, 24.79, 12.92, 26.84, 7.52, 11.79]
}

# 生成数据规模数据
syn_data = {
    'Iteration 1': [120, 56, 54, 33, 28, 28, 171, 16, 9, 10],
    'Iteration 2': [194, 27, 88, 13, 12, 87, 74, 10, 4, 3],
    'Iteration 3': [28, 56, 13, 18, 22, 34, 56, 15, 5, 3]
}

# 图1：箱线图 + 散点
plt.figure(figsize=(12, 7))

# 准备箱线图数据
box_data = [error_data['Iteration 0'], error_data['Iteration 1'], error_data['Iteration 2'], error_data['Iteration 3']]
positions = [0, 1, 2, 3]  # 更新位置为0,1,2,3

# 计算统计量
means = [np.mean(data) for data in box_data]
medians = [np.median(data) for data in box_data]
q1 = [np.percentile(data, 25) for data in box_data]
q3 = [np.percentile(data, 75) for data in box_data]

# 绘制箱线图（不显示异常值，因为我们会用散点显示所有数据）
box_plot = plt.boxplot(box_data, positions=positions, widths=0.5, patch_artist=True,
                      showfliers=False,  # 不显示箱线图的异常值点
                      labels=['Iteration 0', 'Iteration 1', 'Iteration 2', 'Iteration 3'])

# 设置渐变色
colors = ['#037F77','#F1B656', '#397FC7', '#040676']  # 新配色方案
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 设置中位数线为白色并加粗
for median in box_plot['medians']:
    median.set_color('#C94E65')
    median.set_linewidth(3)

# 设置须线和端点样式
for whisker in box_plot['whiskers']:
    whisker.set_color('gray')
    whisker.set_linestyle('--')
    
for cap in box_plot['caps']:
    cap.set_color('gray')

# 添加散点 - 显示所有数据点
scatter_colors = ['#025751', '#D69B35', '#2A5E9A', '#03055E']  # 更深的颜色用于散点，与主色调匹配
for i, (data, pos) in enumerate(zip(box_data, positions)):
    # 添加水平抖动，避免点完全重叠
    x_jitter = np.random.normal(0, 0.08, len(data))
    plt.scatter(np.full(len(data), pos) + x_jitter, 
               data, 
               color=scatter_colors[i], 
               alpha=0.7, 
               s=60,  # 点的大小
               edgecolor='white',
               linewidth=1,
               zorder=3,  # 确保点在最上层
               label=f'Iteration {i+1} Data Points' if i == 0 else "")

# 添加均值线和点
plt.plot(positions, means, 's-', color='orange', linewidth=3, markersize=10, 
         markerfacecolor='white', label='Mean', zorder=4)

# 添加中位数趋势线
plt.plot(positions, medians, '*-', color='#4483B0', linewidth=3, markersize=10, 
         markerfacecolor='white', label='Median', zorder=4)

# 设置坐标轴标签
plt.ylabel('Error (%)', fontsize=24)
plt.xlabel('Iterations', fontsize=24)

# 设置坐标轴刻度标签大小
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# plt.title('Error Distribution with Individual Data Points Across Iterations', 
#           fontsize=16, fontweight='bold', pad=20)
# plt.grid(True, alpha=0.3, axis='y')

# 创建自定义图例
from matplotlib.patches import Patch, Circle
legend_elements = [
    plt.Line2D([0], [0], marker='s', color='orange', markerfacecolor='white', 
               markersize=10, label='Mean', linewidth=3, linestyle='-'),
    plt.Line2D([0], [0], marker='*', color='#4483B0', markerfacecolor='white',
               markersize=14, label='Median', linewidth=3, linestyle='-'),
    # Circle((0, 0), 0.5, facecolor=scatter_colors[0], alpha=0.7, 
    #        edgecolor='white', label='Data Points'),
    Patch(facecolor=colors[0], alpha=0.7, label='Iteration 0 Box'),
    Patch(facecolor=colors[1], alpha=0.7, label='Iteration 1 Box'),
    Patch(facecolor=colors[2], alpha=0.7, label='Iteration 2 Box'),
    Patch(facecolor=colors[3], alpha=0.7, label='Iteration 3 Box')
]

plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1),  # 调整y轴偏移量
                  loc='lower center', fontsize=24, ncol=3)

# 添加统计标注
# for i, (mean_val, median_val) in enumerate(zip(means, medians)):
#     plt.text(positions[i], max(box_data[i]) + 3, 
#              f'Mean: {mean_val:.2f}\nMed: {median_val:.2f}', 
#              ha='center', va='bottom', fontsize=11, fontweight='bold',
#              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

# 设置纵轴范围，上限设为45
plt.ylim(5, 45)  # 设置y轴范围为0到45

# 添加趋势线斜率标注 (Iteration 0到3)
slope_mean = (means[3] - means[0]) / 3
slope_median = (medians[3] - medians[0]) / 3

# plt.text(2, data_max + data_range * 0.15, 
#          f'Mean slope: {slope_mean:.2f} per iteration\nMedian slope: {slope_median:.2f} per iteration', 
#          ha='center', va='bottom', fontsize=11, fontweight='bold',
#          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

plt.tight_layout()
plt.savefig('box_plot.eps', dpi=300, bbox_inches='tight')
plt.savefig('box_plot.png', dpi=300, bbox_inches='tight')

plt.show()


# 图2：堆积柱状图 - SYN数量
plt.figure(figsize=(12, 7))

x_pos = np.arange(len(datasets))
width = 0.8
bottom = np.zeros(len(datasets))
colors = ['#F1B656', '#397FC7', '#040676']

# 定义填充模式
patterns = ['/', 'x', '\\']

# 绘制柱状图
for i, iteration in enumerate(['Iteration 1', 'Iteration 2', 'Iteration 3']):
    bars = plt.bar(x_pos, syn_data[iteration], width, bottom=bottom, 
                  label=iteration, color=colors[i], 
                  edgecolor='black', linewidth=0.5, alpha=0.7)
    # 添加填充模式
    for bar in bars:
        bar.set_hatch(patterns[i])
    bottom += syn_data[iteration]

plt.xlabel('Datasets', fontsize=24)
plt.ylabel('Synthetic Data Size (SYN)', fontsize=24)
# plt.title('Synthetic Data Generation Across Iterations', fontsize=14, fontweight='bold')
plt.xticks(x_pos, datasets, rotation=0, fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('Datasets', fontsize=24)
plt.ylabel('Synthetic Data Size (SYN)', fontsize=24)
plt.legend(bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, fontsize=24)
# plt.grid(True, alpha=0.3, axis='y')

# 添加数据标签
# def add_bar_labels(data_iter1, data_iter2, data_iter3):
#     for i, (d1, d2, d3) in enumerate(zip(data_iter1, data_iter2, data_iter3)):
#         total_height = d1 + d2 + d3
#         # # Iteration 1 标签
#         # if d1 > 0:
#         #     plt.text(i, d1/2, f'{d1}', ha='center', va='center', 
#         #             fontsize=8, fontweight='bold', color='white')
#         # # Iteration 2 标签
#         # if d2 > 0:
#         #     plt.text(i, d1 + d2/2, f'{d2}', ha='center', va='center', 
#         #             fontsize=8, fontweight='bold', color='white')
#         # # Iteration 3 标签
#         # if d3 > 0:
#         #     plt.text(i, d1 + d2 + d3/2, f'{d3}', ha='center', va='center', 
#         #             fontsize=8, fontweight='bold', color='white')
#         # 总计标签
#         # plt.text(i, total_height + 5, f'Total: {total_height}', ha='center', 
#         #         va='bottom', fontsize=8, fontweight='bold', color='black')

# add_bar_labels(syn_data['Iteration 1'], syn_data['Iteration 2'], syn_data['Iteration 3'])
plt.tight_layout()
plt.savefig('iteration_stacked_bar.png', dpi=300, bbox_inches='tight')
plt.savefig('iteration_stacked_bar.eps', dpi=300, bbox_inches='tight')

plt.show()

# 打印详细的趋势分析
print("误差下降趋势分析:")
print("=" * 50)
print(f"Iteration 0 → Iteration 3 误差变化:")
print(f"平均值: {means[0]:.2f} → {means[3]:.2f} (下降: {means[0]-means[3]:.2f}, {((means[0]-means[3])/means[0]*100):.1f}%)")
print(f"中位数: {medians[0]:.2f} → {medians[3]:.2f} (下降: {medians[0]-medians[3]:.2f})")
print(f"每iteration平均下降: {slope_mean:.2f}")
print(f"每iteration中位数下降: {slope_median:.2f}")

# 添加Iteration 0到3的下降分析
print("\nIteration 1 → Iteration 3 误差变化:")
print(f"平均值: {means[1]:.2f} → {means[3]:.2f} (下降: {means[1]-means[3]:.2f}, {((means[1]-means[3])/means[1]*100):.1f}%)")
print(f"中位数: {medians[1]:.2f} → {medians[3]:.2f} (下降: {medians[1]-medians[3]:.2f})")