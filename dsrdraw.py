import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

# # 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [1,1,1,1,1]  # 折线图数据
# subset = [1,1,1,1, 1]   # 柱状图数据
# dataset_name = 'Fodors-Zagats'

# # 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.499,0.499,0.732,0.661,0.499]  # 折线图数据
# subset = [1,1,2,14,1]   # 柱状图数据
# dataset_name = 'Amazon-Google'

# # 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.778,0.919,0.884,0.914,0.900]  # 折线图数据
# subset = [5,5,2,1,1]   # 柱状图数据
# dataset_name = 'DBLP-ACM'

# 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.839,0.839,0.839,0.844,0.839]  # 折线图数据
# subset = [1,1,1,1,1]   # 柱状图数据
# dataset_name = 'DBLP-Scholar'

# 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.875,0.538,0.776,0.623,0.667]  # 折线图数据
# subset = [2,11,5,5,1]   # 柱状图数据
# dataset_name = 'iTunes-Amazon'

# # 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.984,0.980,0.980,0.980,0.976]  # 折线图数据
# subset = [2,1,1,1,1]   # 柱状图数据
# dataset_name = 's-DBLP-ACM'

# 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.886,0.868,0.931,0.930,0.927]  # 折线图数据
# subset = [5,2,1,1,1]   # 柱状图数据
# dataset_name = 's-DBLP-GoogleScholar'

# # 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.935,0.984,0.984,0.984,0.984]  # 折线图数据
# subset = [2,1,1,1,1]   # 柱状图数据
# dataset_name = 's-iTunes-Amazon'

# 示例数据
x = ['0.01', '0.05', '0.1', '0.2', '1']
F1 = [0.734,0.734,0.525,0.805,0.734]  # 折线图数据
subset = [1,1,12,2,1]   # 柱状图数据
dataset_name = 's-Walmart-Amazon'

# # 示例数据
# x = ['0.01', '0.05', '0.1', '0.2', '1']
# F1 = [0.37,0.646,0.521,0.497,0.455]  # 折线图数据
# subset = [5,4,6,4,1]   # 柱状图数据
# dataset_name = 'Walmart-Amazon'

# 创建图表
fig, ax = plt.subplots(figsize=(6, 4))  # 调整图表尺寸（略微减小）

# 设置字体大小
font_multiplier = 2  # 字体放大倍数
label_fontsize = 20 * font_multiplier
tick_fontsize = 18 * font_multiplier
bar_label_fontsize = 16 * font_multiplier  # 柱状图顶部数字字体较小

# 设置字体和标签
ax.set_xlabel(r'$\rho$', fontsize=label_fontsize,labelpad=-0.1)  # 放大x轴标签字体
ax.xaxis.set_label_coords(0.5, -0.36)  # 设置x轴标签的绝对位置（横纵坐标相对轴心）

ax.set_ylabel('NOS', color='black', fontsize=label_fontsize)  # 放大左y轴标签字体
ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)  # 放大刻度标签字体

# 创建柱状图
bars = ax.bar(x, subset, color='white', edgecolor='black', label='NOS', alpha=0.7)
ax.set_ylim(0, 16)
ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))  # 左侧 y 轴只显示少量整数刻度

# 在柱状图上添加数字标注
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height,  # 数字显示位置（略微上移）
            f'{height}', ha='center', va='bottom', fontsize=bar_label_fontsize, color='black')

# 设置横坐标倾斜，确保标签居中
ax.set_xticks(range(len(x)))
ax.set_xticklabels(x, rotation=45, ha='center', fontsize=tick_fontsize)

# 创建右侧y轴（用于F1的参考线）
ax2 = ax.twinx()  # 共享x轴
ax2.set_ylim(0.5, 0.85)
ax2.set_ylabel('F1', fontsize=label_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
from matplotlib.ticker import FormatStrFormatter

# 设置右侧 y 轴刻度格式，保留三位小数
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# 限制右侧y轴刻度为3个主要分割点
ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax2.set_xticks(range(len(x)))
ax2.set_xticklabels(x, fontsize=tick_fontsize)

# 创建折线图
bar_handle = ax.bar(x, subset, color='white', edgecolor='black', label='NOS', alpha=0.7)
line_handle, = ax2.plot(x, F1, color='black', marker='o', markersize=10, markerfacecolor='black',
                        linestyle='-', linewidth=2, label='F1')

# 添加水平线和标记
offset = 0.1
ax2.axhline(y=0.805, color='red', linestyle='--', linewidth=1, label='beyesi F1-score')
ax2.plot(3,0.805, 'rs', markersize=10, markerfacecolor='none', markeredgewidth=2)

# 自定义水平线图例
custom_hline = mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, marker='s',
                             markersize=10, markerfacecolor='none', label='beyesi F1-score')

# 添加数据集名称
plt.text(0.5, 1.06, dataset_name, ha='center', va='bottom', fontsize=label_fontsize, transform=ax.transAxes)

# 调整边缘留白和图表位置
fig.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.18)  # 调整边距以适应放大的字体

# 保存图像到指定文件夹
output_dir = 'plots'  # 指定保存图像的文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果文件夹不存在，则创建
output_file = os.path.join(output_dir, f'{dataset_name}.png')
plt.savefig(output_file, bbox_inches='tight')

# 创建单独的图例
figlegend = plt.figure(figsize=(6, 1))  # 减小图例窗口尺寸
figlegend.legend([bar_handle, line_handle, custom_hline],
                 ['Number Of Subsets (NOS)', 'F1-score (F1)', 'F1 score with Bayesian Optimization'],
                 loc='center', fontsize=12, ncol=3, frameon=False)
figlegend.savefig("legend.png", bbox_inches='tight',dpi=300)
# 显示图例窗口
plt.show()
