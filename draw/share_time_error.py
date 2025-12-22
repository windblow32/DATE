import matplotlib.pyplot as plt
import numpy as np

# 设置Arial字体
plt.rcParams['font.family'] = 'Arial'

# 数据准备
datasets = ['EM', 'EL', 'MT', 'EM', 
           'HE', 'CR', 'CA', 'JA', 'BI', 'BO']

# 无加速策略数据
time_no_accel = [31.79, 7213.52, 1719.13, 7211.41, 301.78, 1649.14, 1883.79, 7205.69, 34, 2809]
error_no_accel = [22.61, 10.51, 17.69, 41.80, 31.98, 27.24, 13.48, 30.45, 7.54, 19]

# 加速策略数据  
time_accel = [27, 1980, 420, 670, 38, 448, 516, 1740, 30, 730.5]
error_accel = [22.98, 13.10, 17.15, 44.03, 34.72, 25.03, 13.72, 27.75, 7.54, 20.79]

# 按照指定顺序排列数据集
target_order = ['MT', 'CA', 'EL', 'BO', 'EM', 'CR', 'HE', 'JA', 'EM', 'BI']
# 创建映射关系来处理重复的数据集名称
original_datasets = ['EM', 'EL', 'MT', 'EM', 
                   'HE', 'CR', 'CA', 'JA', 'BI', 'BO']

# 手动创建索引映射
dataset_mapping = {
    'MT': [2],  # MT在索引2
    'CA': [6],  # CA在索引6
    'EL': [1],  # EL在索引1
    'BO': [9],  # BO在索引9
    'EM_1': [0], # 第一个EM在索引0
    'CR': [5],  # CR在索引5
    'HE': [4],  # HE在索引4
    'JA': [7],  # JA在索引7
    'EM_2': [3], # 第二个EM在索引3
    'BI': [8]   # BI在索引8
}

# 构建排序索引
sorted_indices = []
for dataset_name in target_order:
    if dataset_name == 'EM':
        # 处理两个EM的特殊情况
        if len(sorted_indices) == 4:  # 第一个EM位置
            sorted_indices.append(dataset_mapping['EM_1'][0])
        else:  # 第二个EM位置
            sorted_indices.append(dataset_mapping['EM_2'][0])
    else:
        sorted_indices.append(dataset_mapping[dataset_name][0])

# 重新排列数据
datasets = [datasets[i] for i in sorted_indices]
time_no_accel = [time_no_accel[i] for i in sorted_indices]
time_accel = [time_accel[i] for i in sorted_indices]
error_no_accel = [error_no_accel[i] for i in sorted_indices]
error_accel = [error_accel[i] for i in sorted_indices]

# 计算平均改进
avg_time_no_accel = np.mean(time_no_accel)
avg_time_accel = np.mean(time_accel)
time_reduction = (avg_time_no_accel - avg_time_accel) / avg_time_no_accel * 100
time_speedup = avg_time_no_accel / avg_time_accel

avg_error_no_accel = np.mean(error_no_accel)
avg_error_accel = np.mean(error_accel)
error_change = (avg_error_accel - avg_error_no_accel) / avg_error_no_accel * 100

print(f"平均时间改进: {time_reduction:.1f}% (加速比: {time_speedup:.2f}x)")
print(f"平均误差变化: {error_change:+.1f}%")

# 创建图形和双纵坐标轴
fig, ax1 = plt.subplots(figsize=(14, 8))

# 设置位置和宽度
x = np.arange(len(datasets))
width = 0.35  

# 在左侧y轴绘制柱状图（时间）
bars1 = ax1.bar(x - width/2, time_no_accel, width, label='w/o model sharing: time', color='#C94E65', alpha=0.7, hatch='//', edgecolor='#050301', linewidth=0.5)
bars2 = ax1.bar(x + width/2, time_accel, width, label='model sharing: time', color='#EAB080', alpha=0.7, hatch='\\\\', edgecolor='#050301', linewidth=0.5)

ax1.set_xlabel('Datasets', fontsize=24)
ax1.set_ylabel('Time (s)', fontsize=24, color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=24)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=0, ha='right', fontsize=24)
ax1.set_yscale('log')  # 使用对数坐标因为时间跨度很大

# 创建第二个y轴用于误差
ax2 = ax1.twinx()

# 绘制折线图（误差）
line1 = ax2.plot(x, error_no_accel, color='#C94E64', marker='*', markersize=20, linewidth=4, label='w/o model sharing: error', markerfacecolor='white', markeredgewidth=2)
line2 = ax2.plot(x, error_accel, color='#E79D37', marker='o', markersize=20, linewidth=4, label='model sharing: error', markerfacecolor='none', markeredgewidth=2)

ax2.set_ylabel('Error (%)', fontsize=24, color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=24)

# 组合图例并放在图外正上方居中
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, 
          loc='lower center', bbox_to_anchor=(0.5, 1),
          ncol=2, fancybox=True, shadow=False, frameon=False, fontsize=24)

# 添加改进信息到图表上
# improvement_text = f'Average Time Reduction: {time_reduction:.1f}% ({time_speedup:.1f}x speedup)\nAverage Error Change: {error_change:+.1f}%'
# ax1.text(0.5, 0.95, improvement_text, transform=ax1.transAxes, 
#          fontsize=18, fontweight='bold', ha='center', va='top',
#          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='gray'))
plt.tight_layout()
plt.savefig('share_time_error.eps')
plt.show()