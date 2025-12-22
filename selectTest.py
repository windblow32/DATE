import pandas as pd
import re
from sklearn.model_selection import train_test_split

from ModelShare_with_DSR_final import ModelShare, Rule, Predicate
from generate_prompt import load_json_data

def parse_rule_string(rule_str):
    """Parse a rule string into a set of Predicate objects.
    
    Example input: "(V12 <= 100.0), (V14 > 50.0)"
    Returns: set of Predicate objects
    """
    predicates = set()
    # This regex matches patterns like (V12 <= 100.0)
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, rule_str)
    
    for match in matches:
        # Split into parts: [var, op, value]
        parts = match.strip().split()
        if len(parts) == 3:
            var = parts[0]
            op = parts[1]
            try:
                value = float(parts[2])
                predicates.add(Predicate(var, op, value))
            except ValueError:
                # If value can't be converted to float, skip this predicate
                continue
    return predicates

path = "dataset/clf_num/eye_movements.csv"
# 从path读取dataframe
data = pd.read_csv(path)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 加载block的规则集合
json_path = "current_block/eye_movements_ROU=0.1_f1-error_block.json"
block_data = load_json_data(json_path)

# all_rules 顺序即 block_data.key() 中去除内部字段后的顺序
all_rules = [k for k in block_data.keys() if not k.startswith('_')]
# 合并X_test和y_test为test_data
test_data = pd.concat([X_test, y_test], axis=1)

# ==== 在主程序最前面，先定义一个列表来存储各轮结果 ====
per_round_selected = []   # 用下标与 model_id 对应
# 对于所有的model_id，查看他们覆盖多少数据
for target_id in range(9):
    print(f"\n=== model_id = {target_id} ===")
    
    # 1) 从 test_data 中剔除之前所有轮已选的行
    if per_round_selected:
        # 如果你用列表存，每次把所有已选DataFrame concat后 drop
        all_prev = pd.concat(per_round_selected, ignore_index=False)
        test_data = test_data.drop(all_prev.index, errors='ignore')
    # 如果用字典存，也可以合并 dict.values()

    # 2) 找到本轮所有 target_id 对应的规则
    target_rules = [
        rule for rule in all_rules
        if isinstance(block_data.get(rule), dict)
        and block_data[rule].get('modelID') == target_id
    ]

    # 3) 按规则筛选
    this_round_indices = []
    for rule_str in target_rules:
        preds = parse_rule_string(rule_str)
        rule = Rule(0, set(preds))
        # select_by_rule_for_test 返回 list of lists
        model_share = ModelShare("eye_movements", 0.1)
        selected_list = model_share.select_by_rule_for_test(test_data, rule)
        if not selected_list:
            continue
        # 转回 DataFrame
        selected_df = pd.DataFrame(selected_list, columns=test_data.columns)
        # 定位索引：用 tuple 比对最快
        tup_set = set(selected_df.apply(tuple, axis=1))
        mask = test_data.apply(tuple, axis=1).isin(tup_set)
        idxs = test_data[mask].index.tolist()
        this_round_indices.extend(idxs)

    # 4) 把本轮所有 idx 去重，然后取出对应行
    this_round_indices = list(dict.fromkeys(this_round_indices))  # 保序去重
    this_df = test_data.loc[this_round_indices]

    # 5) 存入 per_round_selected
    per_round_selected.append(this_df)
    # 如果用 dict： per_round_selected[target_id] = this_df

    # 6) 从 test_data 中删除本轮已选行
    test_data = test_data.drop(this_round_indices, errors='ignore')

    # 7) 打印本轮选出规模
    print(f"model_id {target_id}: selected {len(this_df)} rows this round")

# 循环结束后，你就有：
# per_round_selected[i] 是 model_id=i 这一轮选出的 DataFrame
# 也可以合并所有轮： all_selected = pd.concat(per_round_selected, ignore_index=True)

# 最终检查
total_selected = sum(len(df) for df in per_round_selected)
print("\n=== Summary ===")
print(f"Original test set size: {len(X_test)}")
print(f"Total picked across all rounds: {total_selected}")
