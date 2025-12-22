import json
import random
from typing import Dict, List, Tuple, Union, Optional
from unittest import result

import pandas as pd
from openai import OpenAI
from sklearn.model_selection import StratifiedShuffleSplit


def load_json_data(file_path: str) -> Dict:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the parsed JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_rule_and_data(block_data: Dict, block_index: int) -> Tuple[
    Optional[str], Optional[List], Optional[float], Optional[int], Optional[float]]:
    """
    Get the rule, corresponding data, and support value at the specified index.
    
    Args:
        block_data: Dictionary containing the JSON data
        block_index: Index of the rule/data block to retrieve (0-based)
        
    Returns:
        Tuple of (rule, data, support) or (None, None, None) if index is out of range or data is invalid
    """
    if not isinstance(block_data, dict):
        return None, None, None

    rules = [k for k in block_data.keys() if not k.startswith('_')]
    if block_index < 0 or block_index >= len(rules):
        return None, None, None

    rule = rules[block_index]
    rule_data = block_data[rule]

    # Handle different possible structures
    if isinstance(rule_data, dict) and 'data' in rule_data:
        data = rule_data['data']
        support = rule_data.get('support')
        modelID = rule_data.get('modelID')
        validScore = rule_data.get('validScore')
    else:
        data = rule_data
        support = None
        modelID = -1
        validScore = -1

    return rule, data, support, modelID, validScore


def stratified_sample(data, sample_size):
    # 假设 data 是一个列表，其中每个元素也是一个列表，最后一个元素是标签
    df = pd.DataFrame(data)

    # 获取特征和标签
    X = df.iloc[:, :-1]  # 所有列除了最后一列
    y = df.iloc[:, -1]  # 最后一列作为标签

    # 创建 StratifiedShuffleSplit 对象
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)

    # 进行分层抽样
    for train_index, test_index in stratified_split.split(X, y):
        stratified_sample = df.iloc[test_index]

    # 将结果转换为列表形式返回
    return stratified_sample.to_numpy().tolist()


def get_top_n_indices_by_support(block_data: Dict, n: int = 5, target_id: int = 0) -> List[int]:
    """
    从 block_data 中找出 modelID 为 0 或 2 的规则块，
    按照 support * len(data) 从大到小排序，返回前 n 个块在 all_rules 列表中的索引。

    Args:
        block_data: 包含规则块信息的字典
        n: 返回的最大索引数量

    Returns:
        前 n 个符合条件的规则块索引（按出现顺序在 all_rules 的位置）
    """
    if not isinstance(block_data, dict):
        return []

    # all_rules 顺序即 block_data.key() 中去除内部字段后的顺序
    all_rules = [k for k in block_data.keys() if not k.startswith('_')]

    # 收集 (原始索引, 评分) 列表
    indexed_scores = []
    for idx, rule in enumerate(all_rules):
        rule_data = block_data.get(rule)
        # 必须是 dict 且 modelID 在 {0,2} 中
        if not isinstance(rule_data, dict):
            continue
        model_id = rule_data.get('modelID')
        if model_id != target_id and model_id != 0:
            continue

        support = rule_data.get('support', 0.0)
        data_length = len(rule_data.get('data', []))
        score = support * data_length

        indexed_scores.append((idx, score))

    # 按 score 降序排序
    indexed_scores.sort(key=lambda x: x[1], reverse=True)

    # 取前 n 个原始索引
    n = min(len(indexed_scores), n)
    top_n = [idx for idx, _ in indexed_scores[:n]]
    return top_n


def generate_prompt(rules_string: str, data: List, sampleSize=10, modelID=-1, support=0.0, validScore=0.0) -> list[str]:
    """
    Generate a prompt with the given rule and data.
    
    Args:
        rules_string: The rule string
        data: The data associated with the rule
        sampleSize: the number of sampled rows from data
        
    Returns:
        Formatted prompt string
        :param validScore: accuracy
        :param support: 规则支持度
        :param modelID:
        :param rules_string:
        :param data:
        :param sampleSize:
    """
    prompt_parts = []

    # 拆分规则字符串并去除空格
    individual_rules = [rule.strip() for rule in rules_string.split(',')]

    # 翻译规则并输出结果
    translated_rules = [translate_rule(rule) for rule in individual_rules]

    prompt_parts.append(F"Attribute Rule: {translated_rules}")
    prompt_parts.append(F"Focus on: Model {modelID}")
    prompt_parts.append(F"Diversity Score: {support}")
    prompt_parts.append(F"Validation Score: {validScore}")
    prompt_parts.append(
        f"Generated Tabular Data: {stratified_sample(data, sampleSize) if len(data) >= sampleSize else data}\n")
    return prompt_parts


def translate_rule(rule):
    # 解析规则
    attribute, operator, value = rule.strip('()').split()

    # 将操作符翻译为英文
    operator_translation = {
        '>=': '>=',
        '>': '>',
        '<=': '<=',
        '<': '<',
        '==': '==',
        '!=': '!='
    }

    # 构建英文叙述
    if operator in operator_translation:
        english_statement = f"Numerical value of column {attribute} {operator_translation[operator]} {value}."
    else:
        english_statement = "Unknown operator."

    return english_statement


def get_prompt_from_file(data: dict, block_index: int) -> list[str]:
    """
    Get a formatted prompt from a JSON file for the specified block index.
    
    Args:
        data: the JSON file
        block_index: Index of the block to extract (0-based)

    Returns:
        Formatted prompt string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
        IndexError: If block_index is out of range
    """
    # Get the rule, data, and support for the specified index
    rule, block_data, support, modelID, validScore = get_rule_and_data(data, block_index)

    if rule is None or block_data is None:
        raise IndexError(f"Block index {block_index} is out of range.")

    # Generate and return the prompt
    return generate_prompt(rule, block_data)


def get_prompt_from_rule(data: dict, rule: str) -> list[str]:
    """
    Get a formatted prompt for a specific rule.
    
    Args:
        data: The JSON data
        rule: The rule string to get prompt for
        
    Returns:
        Formatted prompt string
    """
    if not isinstance(data, dict) or rule not in data:
        raise ValueError(f"Rule '{rule}' not found in data")

    rule_data = data[rule]
    if isinstance(rule_data, dict) and 'data' in rule_data:
        block_data = rule_data['data']
    else:
        block_data = rule_data

    return generate_prompt(rule, block_data)


def generate_question_strings(json_data, top_n=10, target_id=2):
    """
    Generate question strings from JSON data based on rules with highest support/count ratio.

    Args:
        json_data (dict or list): The loaded JSON data containing rules and their data
        top_n (int): Number of top rules to return (default: 5)

    Returns:
        list: A list of formatted question strings
        modelList: 存储每个问题对应的shareModel
    """
    rules_info = []

    # Convert list to dict if needed
    if isinstance(json_data, list):
        json_data = {str(i): item for i, item in enumerate(json_data)}
    
    # 知识库，用于存储model_D=0以及target_id的规则
    knowledge_base = []
    # Process each rule in the JSON data
    for rule_key, rule_data in json_data.items():
        if isinstance(rule_key, str) and rule_key.startswith('_'):  # Skip metadata
            continue
        # Handle case where rule_data might be a list or a dict
        if isinstance(rule_data, dict):
            # rule_data should be a dict
            support = rule_data.get('support', 0)
            data_list = rule_data.get('data', [])
            modelID = rule_data.get('modelID', 0)
            validScore = rule_data.get('validScore', 0)
        else:  # If rule_data is a list, treat it as the data
            support = 0
            modelID = -1
            data_list = rule_data if isinstance(rule_data, list) else []
            validScore = 0

        # 过滤掉不是0或者target的
        if modelID != target_id and modelID != 0:
            continue
        
        # 除了被选中的top_n，其他满足modelID != target_id and modelID != 0的规则，存储到一个dict中,作为额外知识库

        knowledge_base.append({
            'rule': str(rule_key),
            'support': support,
            'modelID': modelID,
            'data': data_list,
            'validScore': validScore
        })
        
        # 如果候选的question过多，只选取前n个question，如果modelID不是0，就按照support顺序，选取当前ID下前n个question
        if top_n > 10:
            if target_id != 0:
                if modelID != target_id:
                    continue
            
        count = len(data_list)
        support_per_count = support / count if count > 0 else 0
        rules_info.append({
            'rule': str(rule_key),
            'support': support,
            'modelID': modelID,
            'count': count,
            'support_per_count': support_per_count
        })

    # Sort rules by support/count ratio in descending order
    sorted_rules = sorted(rules_info, key=lambda x: x['support_per_count'], reverse=True)

    # Generate question strings for the top rules
    question_strings = []
    modelList = []
    for rule_info in sorted_rules[:top_n]:
        # 拆分规则字符串并去除空格
        individual_rules = [rule.strip() for rule in rule_info['rule'].split(',')]
        # 翻译规则并输出结果
        translated_rules = [translate_rule(rule) for rule in individual_rules]
        question_strings.append(F"Attribute Rule: {translated_rules}")
        modelList.append(rule_info['modelID'])

    return question_strings, modelList, knowledge_base


def promptGenerator(current_block_path: str, dataset_path: str, question_index: int, target_id: int):
    """
    :int question_index: 询问question的rank
    """
    # 数据集变更时修改：label的取值
    classLabel1 = 0
    classLabel2 = 1

    # Example usage
    try:
        # Example 1: Basic usage
        # Load the JSON data
        data = load_json_data(current_block_path)

        # Get top 5 block indices by support
        top_indices = get_top_n_indices_by_support(data, 10, target_id)
        # print(f"Top {len(top_indices)} block indices by support: {top_indices}")

        df = pd.read_csv(dataset_path, nrows=0)
        # 获取DataFrame的列名，即表头
        header = df.columns.tolist()

        # result prompt
        result_prompt: str = ""
        custom_text = "### Your Task ###\n"
        example_prompt = ("Your objective is to generate some rows of tabular data for given Rules. "
                          "You have access to the following attribute Rules with the generated data in the history. "
                          "Each rule is Focus on a Decision Tree model, and can generate a set of high quality tabular data. "
                          "The rules are selected based on the diversity score evaluated with Decision Tree. Rules with higher scores bring more diversity to model and indicate better quality. "
                          f"The generated data and the given example data have the same Attribute list: {header}. "
                          f"The last column of these data is the label({classLabel1} or {classLabel2}), which is the target variable. "
                          "Ensure that the generated data follow the same data type and format as the original table, "
                          "and do not violate any constraints derived from the table's structure."
                          f"For each question, the generated data MUST obey ALL rules in the question, and MUST have the same column as {header}."
                          "For example, rule:[Numerical value of column V10 <= 11.5.], means the column 'V10'(the third column in the header) in the generated data must be less than 11.5.")
        # Print task instructions
        custom_text = custom_text + example_prompt + (
            "Your answer should ONLY include the generated data, and can be extracted into dataframe. For example: [[],[],..,[]]")
        # print(f"{custom_text}\n")
        result_prompt = result_prompt + f"{custom_text}\n"

        # Process top N blocks
        for i, index in enumerate(top_indices, 1):
            try:
                rule, block_data, support, modelID, validScore = get_rule_and_data(data, index)
                # print(f"\n--- Block {i} (Index: {index}, Support: {support if support is not None else 'N/A'}) ---")
                prompt = generate_prompt(rule, block_data, sampleSize=30, modelID=modelID, support=support,
                                         validScore=validScore)
                for row in prompt:
                    result_prompt += row + "\n"

            except Exception as e:
                print(f"Error processing block {index}: {e}")

        # question rule
        result_prompt += "### Question ###\n"
        # 只选取前10个question，如果modelID不是0，就按照support顺序，选取当前ID下前10个question
        question_strings, modelList, knowledge_base = generate_question_strings(data, top_n=question_index + 1,target_id=target_id)
        result_prompt += question_strings[question_index] + "\n"

        result_prompt += f"Focus on: Model {modelList[question_index]}\n"

        result_prompt += "### Answer ###\n"

        result_prompt += "Generated Tabular Data: \n"

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
    except IndexError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return result_prompt, modelList, header, example_prompt, question_strings[question_index], knowledge_base


if __name__ == "__main__":
    json_path = "current_block/bank-marketing_ROU=0.15_f1-error_block.json"
    data_path = "dataset/clf_num/bank-marketing.csv"
    res: str = promptGenerator(current_block_path=json_path, dataset_path=data_path, question_index=0)
    print(res)
