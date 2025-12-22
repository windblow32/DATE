import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from support import compute_rules_fractional_support
from ModelShare_with_DSR_final import Rule
from ModelShare_with_DSR_final import Predicate

from ModelShare_with_DSR_final import Model

'''
用于解决test没有完全被subset划分，导致的大部分test没有model进行验证的问题。
'''


def load_model(dataset, model_id):
    """Load a model from disk using pickle.

    Args:
        model_id: ID of the model to load

    Returns:
        The loaded model or None if not found
        :param model_id: shared model from CRR
        :param dataset: 数据集名称
    """
    filename = f"model/{dataset}_model_{model_id}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_id} not found")
        return None


def load_json_data(file_path: str) -> dict:
    """安全加载JSON数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return {}
    except json.JSONDecodeError:
        print(f"错误: 文件 {file_path} 不是有效的JSON格式")
        return {}


def splitTest(datasetName):
    """分割测试集到不同模型子集"""
    # 1. 加载完整数据集
    path = f"dataset/clf_num/{datasetName}.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"错误: 数据集文件 {path} 不存在")
        return

    # 2. 划分初始训练集和测试集（保留标签）
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1],
        df.iloc[:, -1],
        test_size=0.25,
        random_state=42
    )

    # 合并特征和标签创建完整测试集
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.columns = df.columns  # 保持原始列名

    # 3. 加载规则块数据
    block_path = f"current_block/{datasetName}_ROU=0.2_f1-error_block.json"
    blockData = load_json_data(block_path)
    if not blockData:
        return

    # 4. 按模型ID组织规则
    model_rules = {}
    for rule, data in blockData.items():
        if rule.startswith('_'):  # 忽略元数据
            continue
        modelID = data.get('modelID')
        if modelID not in model_rules:
            model_rules[modelID] = []
        model_rules[modelID].append(rule)

    # 5. 初始化模型子集（空DataFrame）
    model_subsets = {modelID: pd.DataFrame(columns=df.columns) for modelID in model_rules}

    # 6. 分配测试样本
    for idx, row in test_df.iterrows():
        best_model = None
        best_score = -1

        # 寻找最佳匹配模型
        for modelID, rules in model_rules.items():
            # 计算规则支持度
            support_df = compute_rules_fractional_support(row, rules)
            if support_df.empty:
                continue

            # 获取平均支持度
            score = support_df['support_ratio'].mean()
            if score > best_score:
                best_score = score
                best_model = modelID

        # 分配到最佳模型子集
        if best_model is not None:
            # 将行转换为DataFrame并添加
            row_df = pd.DataFrame([row.values], columns=df.columns)
            model_subsets[best_model] = pd.concat([model_subsets[best_model], row_df], ignore_index=True)

    # 7. 保存子集并验证
    total_size = 0
    for modelID, subset in model_subsets.items():
        subset_size = len(subset)
        total_size += subset_size
        save_path = f"model/testset/{datasetName}_subset_{modelID}.csv"
        subset.to_csv(save_path, index=False)
        print(f"模型 {modelID} 的测试子集大小: {subset_size}")

    # 验证总大小
    print(f"\n原始测试集大小: {len(test_df)}")
    print(f"分配后总大小: {total_size}")
    assert total_size == len(test_df), "分配后大小与原始测试集大小不符!"


def remove_rows_merge(df_main: pd.DataFrame, df_remove: pd.DataFrame) -> pd.DataFrame:
    """
    从主DataFrame中移除另一个DataFrame中存在的行
    
    参数:
        df_main: 主DataFrame
        df_remove: 要移除的行所在的DataFrame
        
    返回:
        移除后的主DataFrame
    """
    # 确保列对齐
    common_cols = list(set(df_main.columns).intersection(df_remove.columns))
    
    # 创建临时唯一标识符
    df_main['temp_id'] = df_main[common_cols].apply(tuple, axis=1)
    df_remove['temp_id'] = df_remove[common_cols].apply(tuple, axis=1)
    
    # 过滤不在移除集中的行
    result = df_main[~df_main['temp_id'].isin(df_remove['temp_id'])]
    
    # 移除临时列
    return result.drop(columns=['temp_id'])

def wholeTest(datasetName: str):
    """完整测试集分配"""
    # 1. 加载完整数据集
    path = f"dataset/clf_num/{datasetName}.csv"
    df = pd.read_csv(path)
    
    # 2. 划分初始测试集（保留标签）
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1],
        df.iloc[:, -1],
        test_size=0.25,
        random_state=42
    )
    
    # 创建完整的测试集DataFrame（包含特征和标签）
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.columns = df.columns  # 保持原始列名
    
    # 3. 初始化剩余测试集
    remaining_test = test_df.copy()
    
    # 4. 逐个模型移除已分配的测试数据
    for model_id in range(2):
        print(f"处理模型 {model_id} ...")
        
        # 加载模型数据
        saveData = load_model(datasetName, model_id)
        if saveData is None:
            print(f"模型 {model_id} 未找到，跳过")
            continue
            
        # 获取模型的测试数据
        final_testData = pd.DataFrame(saveData['final_testData'])
        
        # 确保列名一致
        final_testData.columns = test_df.columns
        
        print(f"模型 {model_id} 的测试集大小: {len(final_testData)}")
        
        # 从剩余测试集中移除已分配的行
        remaining_test = remove_rows_merge(remaining_test, final_testData)
        print(f"移除后剩余测试集大小: {len(remaining_test)}\n")
    
    # 5. 保存未分配的测试集
    modelID = -1
    save_path = f"model/testset/{datasetName}_subset_{modelID}.csv"
    remaining_test.to_csv(save_path, index=False)
    print(f"保存未分配测试集到 {save_path}, 大小: {len(remaining_test)}")


# 使用示例
if __name__ == "__main__":
    datasetName = "credit"
    wholeTest(datasetName)
    # splitTest(datasetName)
