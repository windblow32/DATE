import re
import pandas as pd

'''
计算候选规则的支持度，其中每个规则拆成谓词计算
'''
def parse_rule_string(rule_str):
    """
    解析一条规则字符串，返回 [(col, op, value), ...]
    规则形如 "(V14 < 9.0),(V12 >= 607.5)"
    """
    parts = re.findall(r'\([^)]*\)', rule_str)
    preds = []
    for p in parts:
        core = p[1:-1].strip().replace('==', '=')
        tokens = core.split()
        if len(tokens) != 3:
            raise ValueError(f"无法解析谓词: {p}")
        col, op, val = tokens
        # 转数值
        try:
            val = float(val) if '.' in val or 'e' in val.lower() else int(val)
        except:
            val = float(val)
        preds.append((col, op, val))
    return preds


def compute_rules_fractional_support(df: pd.DataFrame, rule_list: list):
    """
    计算每条规则的“部分支持度”：
      对每条样本，满足谓词数/总谓词数 = row_score
      support_ratio = mean(row_score)
      support_sum   = sum(row_score)
    返回 DataFrame 包含 ['rule','support_sum','support_ratio']。
    """
    n = len(df)
    records = []
    for rule_str in rule_list:
        preds = parse_rule_string(rule_str)
        k = len(preds)
        if k == 0:
            # 若无谓词，直接跳过
            records.append({
                'rule': rule_str,
                'support_sum': 0.0,
                'support_ratio': 0.0
            })
            continue

        # 对每个谓词，生成布尔 Series
        mask_sum = pd.Series(0, index=df.index)
        for col, op, val in preds:
            if op == '<':
                mask = df[col] < val
            elif op == '<=':
                mask = df[col] <= val
            elif op == '>':
                mask = df[col] > val
            elif op == '>=':
                mask = df[col] >= val
            elif op in ('=', '=='):
                mask = df[col] == val
            else:
                raise ValueError(f"Unsupported operator '{op}' in {rule_str}")
            # 用 1 表示该谓词是否满足
            mask_sum += mask.astype(int)

        # 每条样本的部分支持度
        row_scores = mask_sum / k
        support_sum = row_scores.sum()  # 总支持度
        support_ratio = support_sum / n if n > 0 else 0.0  # 平均支持度

        records.append({
            'rule': rule_str,
            'support_sum': float(support_sum),
            'support_ratio': float(support_ratio)
        })

    return pd.DataFrame(records, columns=['rule', 'support_sum', 'support_ratio'])


# ===== 使用示例 =====
if __name__ == "__main__":
    # 构造示例 DataFrame
    data = {
        'V12': [600, 610, 620, 590, 700],
        'V14': [8, 10, 5, 12, 7],
        'V1': [1, 2, 3, 4, 5]
    }
    df = pd.DataFrame(data)

    rules = [
        "(V14 < 9.0),(V12 >= 607.5)",
        "(V14 >= 9.0),(V1 < 4)"
    ]

    support_df = compute_rules_fractional_support(df, rules)
    print(support_df)
