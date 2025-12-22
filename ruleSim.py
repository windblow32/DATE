import json
import re
from collections import defaultdict


def parse_rule(rule_str: str) -> dict:
    """
    解析规则字符串为特征条件字典
    
    Args:
        rule_str: 规则字符串，如 "(V12 > 686.5), (V13 <= 24.5)"
        
    Returns:
        字典 {特征: (最小值, 最大值, 原始操作符)}
    """
    # 正则表达式匹配条件
    pattern = r'\((\w+)\s*([<>]=?)\s*([\d.]+)\)'
    conditions = re.findall(pattern, rule_str)

    rule_dict = {}
    for feature, op, value in conditions:
        num_value = float(value)

        # 转换为数值区间
        if op == '>=':
            min_val = num_value
            max_val = float('inf')
        elif op == '>':
            min_val = num_value + 1e-5  # 添加小偏移处理开区间
            max_val = float('inf')
        elif op == '<=':
            min_val = float('-inf')
            max_val = num_value
        elif op == '<':
            min_val = float('-inf')
            max_val = num_value - 1e-5  # 添加小偏移处理开区间
        else:
            continue

        # 存储特征条件（包括原始操作符用于边界处理）
        rule_dict[feature] = (min_val, max_val, op)

    return rule_dict


def interval_overlap(interval1: tuple, interval2: tuple) -> bool:
    """
    判断两个数值区间是否有重叠
    
    Args:
        interval1: (min1, max1)
        interval2: (min2, max2)
        
    Returns:
        bool: 是否有重叠
    """
    min1, max1 = interval1
    min2, max2 = interval2
    return max(min1, min2) < min(max1, max2)


def interval_overlap_ratio(interval1: tuple, interval2: tuple) -> float:
    """
    计算重叠部分占第一个区间的比例
    
    Args:
        interval1: 候选区间 (min1, max1)
        interval2: 测试区间 (min2, max2)
        
    Returns:
        float: 重叠比例 (0.0-1.0)
    """
    min1, max1 = interval1
    min2, max2 = interval2

    # 计算重叠部分
    overlap_min = max(min1, min2)
    overlap_max = min(max1, max2)

    # 没有重叠
    if overlap_min >= overlap_max:
        return 0.0

    # 计算重叠长度和候选长度
    overlap_length = overlap_max - overlap_min
    candidate_length = max1 - min1

    # 处理无限区间
    if candidate_length == float('inf'):
        return 1.0 if overlap_length > 0 else 0.0

    # 处理零长度区间
    if candidate_length == 0:
        return 1.0 if overlap_min == min1 else 0.0

    return overlap_length / candidate_length


def calculate_rule_compatibility(candidate_rule: str,
                                 D_rules: list,
                                 weight_list: list = None) -> tuple:
    """
    计算候选规则与测试集规则的兼容度
    
    Args:
        candidate_rule: 候选规则字符串
        D_rules: 测试集规则字符串列表
        weight_list: 各测试规则的权重列表
        
    Returns:
        (compatibility_score, conflict_score): 兼容度和冲突度分数
    """
    # 默认权重为均匀分布
    if weight_list is None:
        weight_list = [1.0 / len(D_rules)] * len(D_rules)
    elif len(weight_list) != len(D_rules):
        raise ValueError("权重列表长度必须与D_rules相同")

    # 解析候选规则
    candidate_dict = parse_rule(candidate_rule)
    if not candidate_dict:
        return 0.0, 1.0

    # 解析所有测试规则
    parsed_D_rules = [parse_rule(rule) for rule in D_rules]

    # 初始化分数
    total_compatibility = 0.0
    total_conflict = 0.0
    feature_count = len(candidate_dict)

    # 处理每个特征条件
    for feature, (c_min, c_max, c_op) in candidate_dict.items():
        feature_compatibility = 0.0
        feature_conflict = 0.0

        # 检查每个测试规则
        for idx, test_rule in enumerate(parsed_D_rules):
            weight = weight_list[idx]

            # 如果测试规则中没有此特征 - 完全兼容
            if feature not in test_rule:
                feature_compatibility += weight
                continue

            # 获取测试规则的区间条件
            t_min, t_max, t_op = test_rule[feature]
            candidate_interval = (c_min, c_max)
            test_interval = (t_min, t_max)

            # 检查重叠
            if interval_overlap(candidate_interval, test_interval):
                ratio = interval_overlap_ratio(candidate_interval, test_interval)

                # 兼容度计算
                if ratio > 0.9:
                    feature_compatibility += weight
                elif ratio > 0:
                    feature_compatibility += weight * ratio
                else:
                    feature_conflict += weight
            else:
                feature_conflict += weight

        # 累计特征分数
        total_compatibility += feature_compatibility
        total_conflict += feature_conflict

    # 计算平均分数
    avg_compatibility = total_compatibility / feature_count
    avg_conflict = total_conflict / feature_count

    return avg_compatibility, avg_conflict


def calculate_rule_similarity(candidate_rule: str,
                              D_rules: list,
                              weight_list: list = None,
                              conflict_penalty: float = 0.5) -> float:
    """
    计算候选规则与测试集规则的相似度
    
    Args:
        candidate_rule: 候选规则字符串
        D_rules: 测试集规则字符串列表
        weight_list: 各测试规则的权重列表
        conflict_penalty: 冲突惩罚系数
        
    Returns:
        float: 相似度分数 (0.0-1.0)
    """
    comp, conflict = calculate_rule_compatibility(candidate_rule, D_rules, weight_list)
    similarity = max(0.0, comp - conflict_penalty * conflict)
    return similarity


def get_rules_by_model_id(file_path: str, target_model_id: int) -> dict:
    """
    根据指定的 modelID 统计规则及其数据大小
    
    Args:
        file_path: 包含规则块信息的文件路径
        target_model_id: 要筛选的 modelID
        
    Returns:
        字典 {规则字符串: 数据大小}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        block_data = json.load(f)
    if not isinstance(block_data, dict):
        return {}

    # 筛选出所有规则键（排除内部字段）
    rule_keys = [k for k in block_data.keys() if not k.startswith('_')]

    r_list = []
    w_list = []

    for rule in rule_keys:
        rule_data = block_data.get(rule)

        # 跳过非字典项
        if not isinstance(rule_data, dict):
            continue

        # 检查 modelID 是否匹配
        model_id = rule_data.get('modelID')
        if model_id != target_model_id:
            continue

        # 获取数据大小
        data_list = rule_data.get('data', [])
        data_size = len(data_list)

        r_list.append(rule)
        w_list.append(data_size)

    # 归一化放到get_rule_similarity函数中，便于加入新rule时统一计算
    # w_list = [w / sum(w_list) for w in w_list]
    return w_list, r_list


def get_rule_similarity(candidate_rule: str, json_path: str, modelID: int, conflict_penalty: float = 1,
                        new_rules: list = None, new_weights: list = None) -> float:
    '''
    new rules: 被MAB选中的新rule list，需要加入到候选的sim计算集合中
    new_weights: 对应new_rules的数据数量的list，需要归一化形成权重
    '''
    # 读取文件获得权重，weight_list没有归一化
    weight_list, rule_list = get_rules_by_model_id(json_path, modelID)
    # 初始时没有选择新idx，所以weight_list和rule_list为空，此时只需要计算文件中原有的rule
    if new_rules is not None:
        for r in new_rules:
            rule_list.append(r)
        for w in new_weights:
            weight_list.append(w)
        # 根据weight_list的数量，归一化权重
        weight_list = [w / sum(weight_list) for w in weight_list]
    else:
        weight_list = [w / sum(weight_list) for w in weight_list]
        print("new_rules is None")

    similarity = calculate_rule_similarity(candidate_rule, rule_list, weight_list, conflict_penalty=conflict_penalty)
    return similarity
