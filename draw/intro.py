import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn import datasets
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity

# 设置中文字体和风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("white")

def generate_samples_global(X, n_samples=200):
    """整体生成数据 - 学习整体分布"""
    kde = gaussian_kde(X.T)
    generated_samples = kde.resample(n_samples).T
    return generated_samples

def generate_samples_partitioned(X, labels, n_samples=200):
    """划分后生成数据 - 每个分布单独生成"""
    generated_samples = []
    sample_weights = []
    
    # 计算每个分布的样本权重
    for i in np.unique(labels):
        mask = labels == i
        sample_weights.append(np.sum(mask))
    
    sample_weights = np.array(sample_weights) / len(X)
    
    # 为每个分布生成样本
    for i, weight in zip(np.unique(labels), sample_weights):
        mask = labels == i
        cluster_data = X[mask]
        n_cluster_samples = int(n_samples * weight)
        
        if len(cluster_data) > 1:  # 确保有足够数据拟合KDE
            kde = gaussian_kde(cluster_data.T)
            cluster_samples = kde.resample(n_cluster_samples).T
            generated_samples.append(cluster_samples)
    
    return np.vstack(generated_samples)


def plot_generation_comparison(X, labels, colors, names):
    """对比整体生成 vs 划分生成的效果"""
    # 定义不同形状的标记
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'X']
    
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    # 1. 原始数据分布（保持不变）
    ax1 = plt.subplot(gs[0, 0])
    for i in np.unique(labels):
        mask = labels == i
        ax1.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[int(i)], 
                   marker=markers[int(i) % len(markers)],
                   label=names[int(i)],
                   alpha=1, s=50, edgecolors='white', linewidth=0.5)
    # ax1.set_title('1. Original Data Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature X')
    ax1.set_ylabel('Feature Y')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # 2. 整体生成的数据 - 修改：同时显示原始数据和生成数据
    ax2 = plt.subplot(gs[0, 1])
    generated_global = generate_samples_global(X, n_samples=300)
    
    # 先绘制原始数据
    for i in np.unique(labels):
        mask = labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[int(i)], 
                   marker=markers[int(i) % len(markers)],
                   alpha=1, s=30, label=f'Original {names[int(i)]}')
    
    # 再绘制生成数据
    # ax2.scatter(generated_global[:, 0], generated_global[:, 1], 
    #            c='pink', alpha=1, s=50,marker='*', label='Generated Data')
    darker_pink = (0.9, 0.4, 0.5)  # 更深的粉色，RGB值范围0-1
    ax2.scatter(generated_global[:, 0], generated_global[:, 1], 
            c=[darker_pink], alpha=1, s=120, marker='*', label='Generated Data',
            edgecolors='white', linewidth=0.5)

    # ax2.set_title('2. Global Generation\n(No Partition)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature X')
    ax2.set_ylabel('Feature Y')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # 3. 划分后生成的数据 - 修改：同时显示原始数据和生成数据
    ax3 = plt.subplot(gs[0, 2])
    generated_partitioned = generate_samples_partitioned(X, labels, n_samples=300)
    
    # 为生成的样本分配颜色和形状
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    _, indices = nbrs.kneighbors(generated_partitioned)
    generated_labels = labels[indices.flatten()]
    
    # 先绘制原始数据
    for i in np.unique(labels):
        mask = labels == i
        ax3.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[int(i)], 
                   marker=markers[int(i) % len(markers)],
                   alpha=1, s=30, label=f'Original {names[int(i)]}')
    
    # 再绘制生成数据
    # for i in np.unique(labels):
    #     mask = generated_labels == i
    #     if np.sum(mask) > 0:
    #         ax3.scatter(generated_partitioned[mask, 0], generated_partitioned[mask, 1], 
    #                    c=colors[int(i)], 
    #                    marker=markers[int(i) % len(markers)],
    #                    alpha=1, s=50, 
    #                    label=f'Generated {names[int(i)]}')
    # 在绘制生成数据时，使用更深的颜色
    # 在绘制生成数据时，修改蓝色（第二个类别）的加深程度
    for i in np.unique(labels):
        mask = generated_labels == i
        if np.sum(mask) > 0:
            # 将颜色转换为RGB并加深
            color = plt.cm.colors.to_rgb(colors[int(i)])
            # 如果是蓝色（第二个类别），使用更深的颜色
            if i == 1:  # 假设蓝色是第二个类别
                darker_color = tuple([c * 0.4 for c in color])  # 更深的蓝色
            else:
                darker_color = tuple([c * 0.6 for c in color])  # 其他颜色保持原来的加深程度
                
            ax3.scatter(generated_partitioned[mask, 0], generated_partitioned[mask, 1], 
                    c=[darker_color],
                    marker=markers[int(i) % len(markers)],
                    alpha=1, 
                    s=50, 
                    label=f'Generated {names[int(i)]}',
                    linewidth=0.5)
    
    # ax3.set_title('3. Partitioned Generation\n(Per Distribution)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Feature X')
    ax3.set_ylabel('Feature Y')
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = []

    # 添加原始数据图例
    for i in np.unique(labels):
        legend_elements.append(Line2D([0], [0], 
                            marker=markers[int(i) % len(markers)], 
                            color='w', 
                            label=f'Original {names[int(i)]}',
                            markerfacecolor=colors[int(i)],
                            markersize=10,
                            alpha=1))  # 修改为不透明

    # 添加整体生成数据图例
    darker_pink = (0.9, 0.4, 0.5)  # 使用与散点图相同的深粉色
    legend_elements.append(Line2D([0], [0], 
                            marker='*', 
                            color='w', 
                            label='Generated Data (Global)',
                            markerfacecolor=darker_pink,
                            markersize=12,
                            alpha=1))  # 修改为不透明

    # 添加划分生成图例
    # 在创建图例时，对蓝色使用相同的加深逻辑
    for i in np.unique(labels):
        # 使用与散点图相同的颜色加深逻辑
        color = plt.cm.colors.to_rgb(colors[int(i)])
        if i == 1:  # 蓝色类别
            darker_color = tuple([c * 0.3 for c in color])
        else:
            darker_color = tuple([c * 0.6 for c in color])
        
        legend_elements.append(Line2D([0], [0], 
                            marker=markers[int(i) % len(markers)], 
                            color='w', 
                            label=f'Generated {names[int(i)]}',
                            markerfacecolor=darker_color,
                            markersize=10,
                            alpha=1))

    # 调整图形布局，为顶部图例留出空间
    plt.subplots_adjust(top=0.85)

    # 添加图例在图形顶部中央
    fig.legend(handles=legend_elements, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=18)

    plt.tight_layout()
    
    
    
    # # 4. 密度对比 - 整体生成
    # ax4 = plt.subplot(gs[1, 0])
    # # 原始数据密度（低透明度）
    # for i in np.unique(labels):
    #     mask = labels == i
    #     sns.kdeplot(x=X[mask, 0], y=X[mask, 1], 
    #                color=colors[int(i)], alpha=0.3, 
    #                label=f'Original {names[int(i)]}', ax=ax4)
    # # 生成数据密度（高透明度）
    # sns.kdeplot(x=generated_global[:, 0], y=generated_global[:, 1], 
    #            color='red', alpha=0.8, label='Generated Data', ax=ax4)
    # ax4.set_title('4. Density: Global Generation', fontsize=12, fontweight='bold')
    # ax4.set_xlabel('Feature X')
    # ax4.set_ylabel('Feature Y')
    # ax4.legend()
    
    # # 5. 密度对比 - 划分生成
    # ax5 = plt.subplot(gs[1, 1])
    # # 原始数据密度（低透明度）
    # for i in np.unique(labels):
    #     mask = labels == i
    #     sns.kdeplot(x=X[mask, 0], y=X[mask, 1], 
    #                color=colors[int(i)], alpha=0.3, 
    #                label=f'Original {names[int(i)]}', ax=ax5)
    # # 生成数据密度（高透明度）
    # for i in np.unique(labels):
    #     mask = generated_labels == i
    #     if np.sum(mask) > 0:
    #         sns.kdeplot(x=generated_partitioned[mask, 0], y=generated_partitioned[mask, 1], 
    #                    color=colors[int(i)], alpha=0.8, 
    #                    label=f'Generated {names[int(i)]}', ax=ax5)
    # ax5.set_title('5. Density: Partitioned Generation', fontsize=12, fontweight='bold')
    # ax5.set_xlabel('Feature X')
    # ax5.set_ylabel('Feature Y')
    # ax5.legend()
    
    # 6. 质量评估
    ax6 = plt.subplot(gs[1, 2])
    ax6.axis('off')
    
    # 计算质量指标
    def calculate_quality_metrics(original, generated, labels):
        metrics = {}
        
        # 1. 分布一致性（Wasserstein距离）
        from scipy.stats import wasserstein_distance
        metrics['wasserstein_x'] = wasserstein_distance(original[:, 0], generated[:, 0])
        metrics['wasserstein_y'] = wasserstein_distance(original[:, 1], generated[:, 1])
        
        # 2. 覆盖度（生成的样本在原始分布范围内的比例）
        x_range = (original[:, 0].min(), original[:, 0].max())
        y_range = (original[:, 1].min(), original[:, 1].max())
        
        x_in_range = np.sum((generated[:, 0] >= x_range[0]) & (generated[:, 0] <= x_range[1])) / len(generated)
        y_in_range = np.sum((generated[:, 1] >= y_range[0]) & (generated[:, 1] <= y_range[1])) / len(generated)
        metrics['coverage'] = (x_in_range + y_in_range) / 2
        
        # 3. 聚类质量（生成的样本是否能形成清晰的聚类）
        from sklearn.metrics import silhouette_score
        if len(np.unique(labels)) > 1:
            # 为生成数据分配标签
            nbrs = NearestNeighbors(n_neighbors=1).fit(original)
            _, indices = nbrs.kneighbors(generated)
            gen_labels = labels[indices.flatten()]
            metrics['silhouette'] = silhouette_score(generated, gen_labels)
        else:
            metrics['silhouette'] = 0
            
        return metrics
    
    metrics_global = calculate_quality_metrics(X, generated_global, labels)
    metrics_partitioned = calculate_quality_metrics(X, generated_partitioned, labels)
    
    comparison_text = f"""
    🔍 生成质量对比分析：
    
    📊 整体生成 (Global):
    • Wasserstein距离: {metrics_global['wasserstein_x']:.3f} (X), {metrics_global['wasserstein_y']:.3f} (Y)
    • 覆盖度: {metrics_global['coverage']*100:.1f}%
    • 轮廓系数: {metrics_global['silhouette']:.3f}
    
    ⚠️ 问题:
    • 数据位置散乱，缺乏清晰结构
    • 无法学习复杂分布的细节
    • 生成数据分布模糊
    
    📊 划分生成 (Partitioned):
    • Wasserstein距离: {metrics_partitioned['wasserstein_x']:.3f} (X), {metrics_partitioned['wasserstein_y']:.3f} (Y)
    • 覆盖度: {metrics_partitioned['coverage']*100:.1f}%
    • 轮廓系数: {metrics_partitioned['silhouette']:.3f}
    
    ✅ 优势:
    • 数据位置清晰，保持原有结构
    • 准确学习每个分布的独特模式
    • 生成数据分布明确
    """
    
    ax6.text(0.05, 0.95, comparison_text, 
             transform=ax6.transAxes,
             fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', 
                      facecolor='lightblue', 
                      alpha=0.8),
             linespacing=1.6)
    
    plt.tight_layout()
    plt.show()
    
    return generated_global, generated_partitioned

def plot_generation_details(X, labels, colors, names):
    """详细展示生成过程的细节"""
    # 定义不同形状的标记
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'X']
    
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # 生成数据
    generated_global = generate_samples_global(X, n_samples=400)
    generated_partitioned = generate_samples_partitioned(X, labels, n_samples=400)
    
    # 1. 整体生成的散乱问题 - 修改：同时显示原始数据和生成数据
    ax1 = plt.subplot(gs[0, 0])
    
    # 先绘制原始数据
    for i in np.unique(labels):
        mask = labels == i
        ax1.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[int(i)], 
                   marker=markers[int(i) % len(markers)],
                   alpha=1, s=20, label=f'Original {names[int(i)]}')
    
    # 再绘制生成数据
    ax1.scatter(generated_global[:, 0], generated_global[:, 1], 
               c='red', alpha=1, s=40, label='Generated Data')
    
    ax1.set_title('Global Generation: Scattered Data', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature X')
    ax1.set_ylabel('Feature Y')
    ax1.legend()
    # ax1.grid(True, alpha=0.3)
    
    # 添加说明
    ax1.text(0.05, 0.95, "❌ Generated data scattered\nacross all clusters", 
             transform=ax1.transAxes, fontsize=10, color='red',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 2. 划分生成的结构保持 - 修改：同时显示原始数据和生成数据
    ax2 = plt.subplot(gs[0, 1])
    
    # 为生成的样本分配颜色
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(X)
    _, indices = nbrs.kneighbors(generated_partitioned)
    generated_labels = labels[indices.flatten()]
    
    # 先绘制原始数据（低透明度）
    for i in np.unique(labels):
        mask = labels == i
        ax2.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[int(i)], 
                   marker=markers[int(i) % len(markers)],
                   alpha=0.2, s=20, label=f'Original {names[int(i)]}')
    
    # 再绘制生成数据（高透明度）
    for i in np.unique(labels):
        mask = generated_labels == i
        if np.sum(mask) > 0:
            ax2.scatter(generated_partitioned[mask, 0], generated_partitioned[mask, 1], 
                       c=colors[int(i)], 
                       marker=markers[int(i) % len(markers)],
                       alpha=0.8, s=40, 
                       label=f'Generated {names[int(i)]}')
    
    ax2.set_title('Partitioned Generation: Clear Structure', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Feature X')
    ax2.set_ylabel('Feature Y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加说明
    ax2.text(0.05, 0.95, "✅ Generated data maintains\noriginal cluster structure", 
             transform=ax2.transAxes, fontsize=10, color='green',
             bbox=dict(facecolor='white', alpha=0.8))
    
    # 3. 位置分布统计
    ax3 = plt.subplot(gs[1, 0])
    # 计算每个分布的紧凑度（平均最近邻距离）
    def calculate_compactness(data):
        from sklearn.neighbors import NearestNeighbors
        if len(data) < 2:
            return 0
        nbrs = NearestNeighbors(n_neighbors=2).fit(data)
        distances, _ = nbrs.kneighbors(data)
        return np.mean(distances[:, 1])
    
    compactness_global = calculate_compactness(generated_global)
    compactness_partitioned = []
    
    for i in np.unique(labels):
        mask = generated_labels == i
        if np.sum(mask) > 1:
            compactness = calculate_compactness(generated_partitioned[mask])
            compactness_partitioned.append(compactness)
    
    methods = ['Global'] + [f'Partition {i+1}' for i in range(len(compactness_partitioned))]
    compactness_values = [compactness_global] + compactness_partitioned
    
    bars = ax3.bar(methods, compactness_values, 
                   color=['red'] + colors[:len(compactness_partitioned)],
                   alpha=0.7)
    ax3.set_title('Data Compactness Comparison', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Nearest Neighbor Distance')
    ax3.set_xlabel('Generation Method')
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, compactness_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    

def plot_custom_heterogeneous_data():
    """绘制自定义异构数据并展示生成效果对比"""
    # 设置随机种子
    np.random.seed(42)
    
    # 创建有明显差异的分布
    # 分布1: 紧凑的圆形分布
    theta = np.random.uniform(0, 2*np.pi, 80)
    r = np.random.normal(2, 0.2, 80)
    cluster1 = np.column_stack([r * np.cos(theta) + 1, r * np.sin(theta) + 1])
    
    # 分布2: 分散的线性分布
    x2 = np.random.uniform(-2, 4, 100)
    y2 = 0.6 * x2 + 1 + np.random.normal(0, 0.8, 100)
    cluster2 = np.column_stack([x2, y2])
    
    # 分布3: 另一个紧凑分布
    theta3 = np.random.uniform(0, 2*np.pi, 70)
    r3 = np.random.normal(1.5, 0.25, 70)
    cluster3 = np.column_stack([r3 * np.cos(theta3) - 2, r3 * np.sin(theta3) - 1])
    
    # 合并数据
    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.hstack([np.zeros(80), np.ones(100), np.ones(70)*2])
    
    # 定义颜色和名称
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    distribution_names = ['Distribution A', 'Distribution B', 'Distribution C']
    
    print("Generating data and comparing methods...")
    print("=" * 50)
    
    # 1. 主要对比图
    generated_global, generated_partitioned = plot_generation_comparison(X, labels, colors, distribution_names)
    
    # 2. 详细分析图
    # plot_generation_details(X, labels, colors, distribution_names)
    
    return X, labels, generated_global, generated_partitioned

# 运行绘图函数
if __name__ == "__main__":
    print("展示整体生成 vs 划分生成的效果对比...")
    X_custom, labels_custom, global_data, partitioned_data = plot_custom_heterogeneous_data()
    
    # 打印统计信息
    print(f"\n原始数据形状: {X_custom.shape}")
    print(f"整体生成数据形状: {global_data.shape}")
    print(f"划分生成数据形状: {partitioned_data.shape}")
    print(f"原始数据类别分布: {[np.sum(labels_custom == i) for i in np.unique(labels_custom)]}")