import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('cd_DATE.csv')

# 清理数据 - 移除空行
df = df.dropna(subset=['accuracy'])

# 获取所有唯一的方法名称
methods = df['classifier_name'].unique()

# 获取所有唯一的数据集名称
datasets = df['dataset_name'].unique()

# 创建一个矩阵来存储每个方法在每个数据集上的准确率
accuracy_matrix = np.zeros((len(methods), len(datasets)))

for i, method in enumerate(methods):
    method_data = df[df['classifier_name'] == method]
    for j, dataset in enumerate(datasets):
        dataset_data = method_data[method_data['dataset_name'] == dataset]
        if not dataset_data.empty:
            accuracy_matrix[i, j] = dataset_data['accuracy'].values[0]
        else:
            accuracy_matrix[i, j] = np.nan

# 进行Wilcoxon符号秩检验
n_methods = len(methods)
p_value_matrix = np.ones((n_methods, n_methods))

for i in range(n_methods):
    for j in range(n_methods):
        if i != j:
            # 获取两个方法在所有数据集上的准确率
            acc_i = accuracy_matrix[i, :]
            acc_j = accuracy_matrix[j, :]
            
            # 移除NaN值
            mask = ~(np.isnan(acc_i) | np.isnan(acc_j))
            acc_i_clean = acc_i[mask]
            acc_j_clean = acc_j[mask]
            
            if len(acc_i_clean) > 0:
                try:
                    # 执行Wilcoxon检验
                    stat, p_value = wilcoxon(acc_i_clean, acc_j_clean)
                    p_value_matrix[i, j] = p_value
                except:
                    p_value_matrix[i, j] = 1.0

# 创建热力图 - 使用蓝色系配色方案
plt.figure(figsize=(12, 10))

# 创建mask来隐藏对角线
mask = np.eye(n_methods, dtype=bool)

# 自定义蓝色系配色方案 - 从浅蓝到深蓝
blue_cmap = sns.color_palette(["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", 
                              "#42A5F5", "#2196F3", "#1E88E5", "#1976D2", 
                              "#1565C0", "#0D47A1"], as_cmap=True)
# blue_cmap = sns.color_palette(["#0D47A1", "#1565C0", "#1976D2", "#1E88E5", "#2196F3", 
#  "#42A5F5", "#64B5F6", "#90CAF9", "#BBDEFB", "#E3F2FD"], as_cmap=True)


# # 绘制热力图
heatmap = sns.heatmap(p_value_matrix, 
            xticklabels=methods,
            yticklabels=methods,
            annot=True, 
            fmt='.3f',
            cmap=blue_cmap,
            cbar_kws={'label': 'p-value', 'shrink': 0.8},
            mask=mask,
            square=True,
            annot_kws={'size': 14})

# 绘制热力图
# heatmap = sns.heatmap(p_value_matrix, 
#             xticklabels=methods,
#             yticklabels=methods,
#             annot=True, 
#             fmt='.3f',
#             cmap=blue_cmap.reverse(),  # 反转颜色映射，使更小的p值颜色更深
#             cbar_kws={'label': 'p-value (darker = more significant)', 'shrink': 0.8},
#             mask=mask,
#             square=True,
#             annot_kws={'size': 14},
#             vmin=0,  # 确保颜色映射从0开始
#             vmax=1)  # 到1结束

# 增加颜色条字体大小
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)  # 刻度字体大小
cbar.set_label('p-value', fontsize=20)  # 标签字体大小

# 找到DATE方法的位置并高亮显示最显著的DATE
date_indices = [i for i, method in enumerate(methods) if 'DATE' in method]
if date_indices:
    # 找到最显著的DATE方法（与最多其他方法有显著差异的）
    most_significant_date_idx = None
    max_significant_count = 0
    
    for idx in date_indices:
        significant_count = sum(1 for j in range(len(methods)) 
                              if idx != j and p_value_matrix[idx, j] < 0.05)
        if significant_count > max_significant_count:
            max_significant_count = significant_count
            most_significant_date_idx = idx
    
    if most_significant_date_idx is not None:
        # 高亮显示最显著的DATE方法的行和列标签
        method_name = methods[most_significant_date_idx]
        
        # 重新绘制热力图，高亮最显著的DATE
        ax = plt.gca()
        
        # 为最显著的DATE方法添加红色边框
        for i, text in enumerate(ax.get_xticklabels()):
            if i == most_significant_date_idx:
                text.set_color('red')
                text.set_fontweight('bold')
                
        for i, text in enumerate(ax.get_yticklabels()):
            if i == most_significant_date_idx:
                text.set_color('red')
                text.set_fontweight('bold')

# plt.title('Wilcoxon Signed-Rank Test: p-values for Pairwise Comparisons', 
#           fontsize=14, fontweight='bold', pad=20)
# plt.xlabel('Methods', fontsize=16)
# plt.ylabel('Methods', fontsize=16)
plt.xticks(rotation=30, ha='right', fontsize=20)
plt.yticks(rotation=0, fontsize=20)

# 调整布局
plt.tight_layout()
plt.savefig('wilcoxon_heatmap_blue.png', dpi=300, bbox_inches='tight')
plt.show()

# 显著性标记的热力图 - 使用双色蓝色系
plt.figure(figsize=(12, 10))

# # 创建自定义的双色蓝色配色
# from matplotlib.colors import ListedColormap


# # 可选：创建更精细的配色方案（类似图片中的多级蓝色）
# plt.figure(figsize=(12, 10))

# # 使用多级蓝色配色 - 从浅蓝到深蓝
# multi_blue_cmap = sns.color_palette(["#E3F2FD", "#BBDEFB", "#90CAF9", "#64B5F6", 
#                                     "#42A5F5", "#2196F3", "#1E88E5", "#1976D2", 
#                                     "#1565C0", "#0D47A1"], as_cmap=True)

# sns.heatmap(p_value_matrix, 
#             xticklabels=methods,
#             yticklabels=methods,
#             annot=True, 
#             fmt='.3f',
#             cmap=multi_blue_cmap,
#             cbar_kws={'label': 'p-value', 'shrink': 0.8},
#             mask=mask,
#             square=True,
#             annot_kws={'size': 14})

# # 增加颜色条字体大小
# cbar = plt.gca().collections[0].colorbar
# cbar.ax.tick_params(labelsize=16)  # 刻度字体大小
# cbar.set_label('p-value', fontsize=18)  # 标签字体大小

# # 找到DATE方法的位置并高亮显示最显著的DATE
# date_indices = [i for i, method in enumerate(methods) if 'DATE' in method]
# if date_indices:
#     # 找到最显著的DATE方法（与最多其他方法有显著差异的）
#     most_significant_date_idx = None
#     max_significant_count = 0
    
#     for idx in date_indices:
#         significant_count = sum(1 for j in range(len(methods)) 
#                               if idx != j and p_value_matrix[idx, j] < 0.05)
#         if significant_count > max_significant_count:
#             max_significant_count = significant_count
#             most_significant_date_idx = idx
    
#     if most_significant_date_idx is not None:
#         # 高亮显示最显著的DATE方法的行和列标签
#         method_name = methods[most_significant_date_idx]
        
#         # 重新绘制热力图，高亮最显著的DATE
#         ax = plt.gca()
        
#         # 为最显著的DATE方法添加红色边框
#         for i, text in enumerate(ax.get_xticklabels()):
#             if i == most_significant_date_idx:
#                 text.set_color('red')
#                 text.set_fontweight('bold')
                
#         for i, text in enumerate(ax.get_yticklabels()):
#             if i == most_significant_date_idx:
#                 text.set_color('red')
#                 text.set_fontweight('bold')

# # plt.title('Wilcoxon Signed-Rank Test: p-values for Pairwise Comparisons', 
# #           fontsize=14, fontweight='bold', pad=20)
# # plt.xlabel('Methods', fontsize=16)
# # plt.ylabel('Methods', fontsize=16)
# plt.xticks(rotation=45, ha='right', fontsize=14)
# plt.yticks(rotation=0, fontsize=14)

# plt.tight_layout()
# plt.savefig('wilcoxon_heatmap_multi_blue.png', dpi=300, bbox_inches='tight')
# plt.show()

# 打印统计摘要
print("Wilcoxon Signed-Rank Test 统计摘要:")
print("=" * 50)
for i in range(n_methods):
    for j in range(i+1, n_methods):
        p_val = p_value_matrix[i, j]
        significance = "显著" if p_val < 0.05 else "不显著"
        print(f"{methods[i]} vs {methods[j]}: p-value = {p_val:.4f} ({significance})")
