import matplotlib.pyplot as plt
import numpy as np

# 设置Arial字体和24号字体大小
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24

# 设置专业图表样式
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 数据准备
rou = [0.1, 0.075, 0.05, 0.025, 0.01]

# EL 数据集
EL_metric1 = [13.23, 13.17, 13.10, 12.69, 12.36]
EL_metric2 = [13.08, 12.82, 12.78, 11.55, 11.98]
EL_time = [1595.97, 1899.62, 1980.00, 2110.39, 2365.22]

# MT 数据集  
MT_metric1 = [20.87, 20.87, 17.15, 17.71, 42.19]
MT_metric2 = [20.05, 20.05, 16.58, 16.55, 40.35]
MT_time = [308.82, 393.46, 420.00, 438.18, 455.35]

# CA 数据集
CA_metric1 = [15.52, 13.91, 13.72, 12.55, 12.20]
CA_metric2 = [14.92, 13.58, 12.92, 12.27, 12.03]
CA_time = [440.02, 464.78, 516.00, 519.00, 547.17]

# 第一行：metric1 和 metric2 对比
datasets = ['EL', 'MT', 'CA']
metric1_data = [EL_metric1, MT_metric1, CA_metric1]
metric2_data = [EL_metric2, MT_metric2, CA_metric2]

# 收集所有图例句柄和标签
legend_handles = []
legend_labels = []

for i, dataset in enumerate(datasets):
    line1, = axes[0,i].plot(rou, metric1_data[i], 's-', label='DATE (w/o generation)', 
                            linewidth=2, markersize=8, color='#1f77b4')
    line2, = axes[0,i].plot(rou, metric2_data[i], 'o-', label='DATE', 
                            linewidth=2, markersize=8, color='#ff7f0e')
    
    # 只在第一个子图收集图例信息
    if i == 0:
        legend_handles.extend([line1, line2])
        legend_labels.extend(['DATE (w/o generation)', 'DATE'])
    
    axes[0,i].set_xlabel('$\\rho$', fontsize=24)
    axes[0,i].set_ylabel('Error Rate (%)', fontsize=24)
    axes[0,i].set_title(f'{dataset}: Error Comparison', fontsize=24, fontweight='bold')
    axes[0,i].grid(True, alpha=0.3)
    axes[0,i].tick_params(axis='both', labelsize=24)
    # 反转x轴，因为rou值从大到小
    axes[0,i].invert_xaxis()

# 第二行：运行时间分析
time_data = [EL_time, MT_time, CA_time]
for i, dataset in enumerate(datasets):
    line3, = axes[1,i].plot(rou, time_data[i], '^-', color='#2ca02c', linewidth=2, markersize=8, label='Running Time')
    
    # 只在第一个子图收集时间图例信息
    if i == 0:
        legend_handles.append(line3)
        legend_labels.append('Running Time')
    
    axes[1,i].set_xlabel('$\\rho$', fontsize=24)
    axes[1,i].set_ylabel('Time (s)', fontsize=24)
    axes[1,i].set_title(f'{dataset}: Running Time', fontsize=24, fontweight='bold')
    axes[1,i].grid(True, alpha=0.3)
    axes[1,i].tick_params(axis='both', labelsize=24)
    axes[1,i].invert_xaxis()

# 在图底部添加统一图例
fig.legend(legend_handles, legend_labels, loc='lower center', 
           bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=24, 
           frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
# 调整底部边距为图例留出空间
plt.subplots_adjust(bottom=0.08)
plt.show()