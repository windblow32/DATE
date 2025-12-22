import json
import os

def merge_blocks_with_rules(csv_file, json_file, output_file):
    # 读取JSON文件中的规则和数据
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # 创建数据到规则的映射
    data_to_rule = {}
    for rule, data_list in json_data.items():
        # 将每个数据项转换为字符串形式作为键
        for data in data_list:
            data_str = ','.join(map(str, data))
            data_to_rule[data_str] = rule
    
    # 存储所有modelID的数据和规则
    model_info = {}
    
    # 读取CSV文件
    with open(csv_file, 'r') as f:
        current_model_id = None
        current_data = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('modelID'):
                # 保存前一个modelID的数据
                if current_model_id is not None and current_data:
                    # 查找对应的规则
                    first_data_str = ','.join(map(str, current_data[0]))
                    rule = data_to_rule.get(first_data_str, "No matching rule found")
                    model_info[current_model_id] = {
                        'rule': rule,
                        'data': current_data
                    }
                    current_data = []
                
                # 更新当前modelID
                current_model_id = line.split(':')[1].strip()
            else:
                # 处理数据行
                try:
                    data = [float(x) for x in line.split(',')]
                    current_data.append(data)
                except ValueError:
                    print(f"Warning: Skipping invalid line: {line}")
        
        # 处理最后一个block
        if current_model_id is not None and current_data:
            first_data_str = ','.join(map(str, current_data[0]))
            rule = data_to_rule.get(first_data_str, "No matching rule found")
            model_info[current_model_id] = {
                'rule': rule,
                'data': current_data
            }
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        for model_id, info in model_info.items():
            f.write(f"modelID : {model_id}\n")
            f.write(f"Rule: {info['rule']}\n")
            # 写入数据
            for data in info['data']:
                f.write(','.join(map(str, data)) + '\n')
            f.write('\n')  # 添加空行分隔不同的blocks
    
    print(f"处理完成！输出文件：{output_file}")
    
    # 打印一些统计信息
    print(f"\n统计信息：")
    print(f"总共处理了 {len(model_info)} 个不同的modelID")
    rules_count = {}
    for info in model_info.values():
        rule = info['rule']
        rules_count[rule] = rules_count.get(rule, 0) + 1
    print("\n规则统计：")
    for rule, count in rules_count.items():
        print(f"{rule}: {count}个modelID")

if __name__ == '__main__':
    # 文件路径
    csv_file = 'block/bank-marketing_ROU=0.21_f1-error_block.csv'
    json_file = 'current_block/bank-marketing_ROU=0.21_f1-error_block.json'
    output_file = 'merged_blocks_with_rules.csv'
    
    # 执行合并
    merge_blocks_with_rules(csv_file, json_file, output_file)
