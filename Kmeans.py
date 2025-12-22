import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os


def rule_based_clustering(csv_file, n_clusters=3, max_depth=4, random_state=42):
    """
    基于规则归纳的聚类方法
    
    参数:
    csv_file: 输入的CSV文件路径
    n_clusters: 聚类数量
    max_depth: 决策树最大深度，控制规则复杂度
    random_state: 随机种子
    """

    # 1. 读取数据
    print("步骤1: 读取数据...")
    data = pd.read_csv(csv_file)
    print(f"数据形状: {data.shape}")
    print(f"特征列: {list(data.columns)}")

    # 2. 数据预处理
    print("\n步骤2: 数据预处理...")
    # 选择数值型列
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    print(f"数值型特征: {numeric_columns}")

    if len(numeric_columns) == 0:
        raise ValueError("未找到数值型特征列，请确保CSV文件中包含数值数据")

    X = data[numeric_columns].copy()

    # 处理缺失值
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        print(f"发现缺失值: {missing_values[missing_values > 0]}")
        # 用中位数填充缺失值
        X = X.fillna(X.median())

    # 标准化数据（对聚类很重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = numeric_columns

    # 3. 聚类分析
    print("\n步骤3: 进行K-means聚类...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 将聚类结果添加到原始数据中
    data['cluster'] = cluster_labels
    X['cluster'] = cluster_labels

    # 4. 规则提取和验证
    print("\n步骤4: 提取聚类规则并验证...")
    # 使用标准化前的数据训练决策树，便于解释
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X[feature_names], cluster_labels)

    # 预测并计算准确率
    y_pred = dt.predict(X[feature_names])
    accuracy = accuracy_score(cluster_labels, y_pred)

    # 生成规则文本并解析
    rules_text = export_text(dt, feature_names=feature_names)
    print("\n决策树规则:")
    print(rules_text)

    # 5. 构建JSON结果
    print("\n步骤5: 构建JSON结果...")
    result_json = {}

    # 解析决策树规则，获取每个叶节点对应的规则和聚类
    leaf_info = parse_decision_tree_rules(rules_text, feature_names)

    total_samples = len(data)

    for rule, cluster_id in leaf_info.items():
        cluster_data = data[data['cluster'] == cluster_id]
        cluster_indices = data[data['cluster'] == cluster_id].index

        # 计算该规则下的预测准确率（只针对这个聚类）
        cluster_mask = (cluster_labels == cluster_id)
        cluster_accuracy = accuracy_score(cluster_labels[cluster_mask], y_pred[cluster_mask])

        # 获取该聚类的数据（转换为列表格式）
        cluster_data_list = X.loc[cluster_indices, feature_names].values.tolist()

        # 构建结果
        result_json[rule] = {
            "modelID": int(cluster_id),
            "support": round(len(cluster_data) / total_samples, 4),
            "validScore": round(cluster_accuracy, 4),
            "data": cluster_data_list
        }

    # 6. 保存结果
    print("\n步骤6: 保存结果...")

    # 创建结果目录
    output_dir = 'clustering_results'
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON结果
    json_output_path = f'{output_dir}/' + datasetName + '.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    # 保存完整的带聚类标签的数据
    data.to_csv(f'{output_dir}/clustered_data.csv', index=False)

    # 为每个聚类创建单独的文件
    for cluster_id in range(n_clusters):
        cluster_data = data[data['cluster'] == cluster_id]
        cluster_data.to_csv(f'{output_dir}/cluster_{cluster_id}_data.csv', index=False)

    # 7. 生成摘要报告
    print("\n步骤7: 生成摘要报告...")
    print("\n" + "=" * 60)
    print("聚类结果摘要")
    print("=" * 60)

    for rule, info in result_json.items():
        print(f"\n规则: {rule}")
        print(f"  聚类ID: {info['modelID']}")
        print(f"  支持度: {info['support']:.4f} ({int(info['support'] * total_samples)}/{total_samples})")
        print(f"  准确率: {info['validScore']:.4f}")
        print(f"  数据点数: {len(info['data'])}")

    print(f"\n总体准确率: {accuracy:.4f}")
    print("=" * 60)

    return result_json, data, accuracy


def parse_decision_tree_rules(rules_text, feature_names):
    """
    解析决策树规则文本，返回每个叶节点的规则路径和对应的聚类ID
    规则格式改为: (feature1_name < xxx),(feature2_name < xxx),...
    """
    leaf_info = {}
    lines = rules_text.split('\n')

    current_rule_parts = []  # 存储规则部分
    rule_stack = []  # 用于跟踪规则路径

    for line in lines:
        if not line.strip():
            continue

        # 计算当前行的深度
        depth = len(line) - len(line.lstrip())
        indent = depth // 3  # sklearn的export_text每层缩进3个空格

        line = line.strip()

        if 'class: ' in line:
            # 叶节点行
            try:
                # 尝试提取类标签，处理不同格式：
                # 1. class: cluster_X
                # 2. class: X
                class_part = line.split('class: ')[1].strip()
                if '_' in class_part:
                    cluster_id = int(class_part.split('_')[1])
                else:
                    # 如果格式是直接的数字
                    cluster_id = int(class_part)

                # 构建规则字符串：用逗号分隔所有条件
                if current_rule_parts:
                    full_rule = ",".join(current_rule_parts)
                else:
                    full_rule = "True"  # 如果没有条件，表示根节点

                leaf_info[full_rule] = cluster_id

                # 重置当前规则部分，准备下一个路径
                current_rule_parts = []
            except (IndexError, ValueError) as e:
                print(f"警告: 无法解析行: {line.strip()}")
                print(f"错误详情: {str(e)}")
                continue

        elif '---' in line:
            # 决策节点行
            condition = line.split('--- ')[1]

            # 调整规则栈以匹配当前深度
            while len(rule_stack) > indent:
                rule_stack.pop()

            if len(rule_stack) == indent:
                rule_stack.append(condition)
            else:
                rule_stack[indent] = condition

            # 将条件转换为要求的格式
            formatted_condition = format_condition(condition)
            if formatted_condition:
                # 更新当前规则部分
                if len(current_rule_parts) <= indent:
                    current_rule_parts.append(formatted_condition)
                else:
                    current_rule_parts[indent] = formatted_condition
                # 移除更深层的条件
                current_rule_parts = current_rule_parts[:indent + 1]

    return leaf_info


def format_condition(condition):
    """
    将决策树条件转换为 (feature_name operator value) 格式
    """
    # 处理条件字符串
    condition = condition.strip()

    # 匹配模式：特征名 操作符 值
    parts = condition.split()
    if len(parts) < 3:
        return None

    feature_name = parts[0]
    operator = parts[1]
    value = ' '.join(parts[2:])

    # 标准化操作符显示
    operator_map = {
        '<=': '<',
        '>=': '>=',
        '<': '<',
        '>': '>'
    }

    if operator in operator_map:
        formatted_operator = operator_map[operator]
        # 尝试将值转换为更简洁的格式
        try:
            # 如果是数值，保留适当的小数位数
            value_float = float(value)
            if value_float == int(value_float):
                value_str = str(int(value_float))
            else:
                value_str = f"{value_float:.4f}".rstrip('0').rstrip('.')
        except ValueError:
            value_str = value

        return f"({feature_name} {formatted_operator} {value_str})"

    return None


def visualize_clustering_results(result_json, output_dir='clustering_results'):
    """
    可视化聚类结果
    """
    # 创建可视化
    plt.figure(figsize=(12, 8))

    # 准备数据用于绘图
    supports = []
    scores = []
    labels = []

    for rule, info in result_json.items():
        supports.append(info['support'])
        scores.append(info['validScore'])
        labels.append(f"Cluster {info['modelID']}")

    # 绘制支持度和准确率的条形图
    x = np.arange(len(result_json))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('聚类')
    ax1.set_ylabel('支持度', color=color)
    bars1 = ax1.bar(x - width / 2, supports, width, label='支持度', color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('准确率', color=color)
    bars2 = ax2.bar(x + width / 2, scores, width, label='准确率', color=color, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('聚类结果统计')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    fig.tight_layout()

    # 添加数值标签
    for i, (support, score) in enumerate(zip(supports, scores)):
        ax1.text(i - width / 2, support + 0.01, f'{support:.3f}', ha='center', va='bottom')
        ax2.text(i + width / 2, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

    plt.savefig(f'{output_dir}/clustering_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


# 使用示例
if __name__ == "__main__":
    datasetName = "bank-marketing"
    dataPath = "dataset/clf_num/bank-marketing.csv"
    try:
        # 执行聚类分析
        result_json, result_data, overall_accuracy = rule_based_clustering(
            csv_file=dataPath,  # 替换为您的CSV文件路径
            n_clusters=3,  # 聚类数量，可根据数据调整
            max_depth=4,  # 规则复杂度，建议3-5
            random_state=42
        )

        # 生成可视化
        # visualize_clustering_results(result_json)

        # 打印最终的JSON结果
        print("\n最终JSON结果:")
        print(json.dumps(result_json, indent=2, ensure_ascii=False))

        print(f"\n处理完成！结果已保存到 clustering_results 目录")
        print(f"总体准确率: {overall_accuracy:.4f}")

    except FileNotFoundError:
        print("文件未找到，请检查文件路径")
        print("\n请按以下步骤使用:")
        print("1. 将 'your_data.csv' 替换为您的实际CSV文件路径")
        print("2. 确保CSV文件包含数值型特征列")
        print("3. 根据需要调整 n_clusters 和 max_depth 参数")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
