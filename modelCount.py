import json


def load_json_data(file_path: str):
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


def count(current_block_path):
    data = load_json_data(current_block_path)
    if not isinstance(data, dict):
        return []

    # all_rules 顺序即 block_data.key() 中去除内部字段后的顺序
    all_rules = [k for k in data.keys() if not k.startswith('_')]

    # 统计所有share相同modelID的规则数量
    modelID_count = {}
    for idx, rule in enumerate(all_rules):
        rule_data = data.get(rule)
        # 必须是 dict 且 modelID 在 {0,2} 中
        if not isinstance(rule_data, dict):
            continue
        model_id = rule_data.get('modelID')
        # 为不同modelID的subset分别计数
        if model_id not in modelID_count:
            modelID_count[model_id] = 0
        modelID_count[model_id] += 1
    return modelID_count


datasetName = "credit"
json_path = "current_block/" + datasetName + "_ROU=0.2_f1-error_block.json"
res = count(json_path)
print(res)
