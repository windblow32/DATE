import pandas as pd
import os

def merge_blocks(input_dir):
    # 存储所有modelID的数据
    model_data = {}
    
    # 遍历block文件夹下的所有csv文件
    file_path = 'block/bank-marketing_ROU=0.2_f1-error_block.csv'
    current_model_id = None
    current_data = []
            
    with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('modelID'):
                        # 如果遇到新的modelID，保存之前的数据
                        if current_model_id is not None and current_data:
                            if current_model_id not in model_data:
                                model_data[current_model_id] = []
                            model_data[current_model_id].extend(current_data)
                            current_data = []
                        # 更新当前modelID
                        current_model_id = line.split(':')[1].strip()
                    else:
                        # 将数据行添加到当前modelID的数据中
                        if current_model_id is not None and line:
                            current_data.append(line.split(','))
                
                # 处理文件最后一个block的数据
                if current_model_id is not None and current_data:
                    if current_model_id not in model_data:
                        model_data[current_model_id] = []
                    model_data[current_model_id].extend(current_data)
    
    # 将合并后的数据写入新文件
    output_file = os.path.join(input_dir, 'merged_blocks.csv')
    with open(output_file, 'w') as f:
        for model_id, data in model_data.items():
            f.write(f'modelID : {model_id}\n')
            for row in data:
                f.write(','.join(row) + '\n')
            f.write('\n')  # 在不同modelID的数据块之间添加空行
    
    print(f'合并完成，输出文件：{output_file}')

if __name__ == '__main__':
    block_dir = os.path.join(os.path.dirname(__file__), 'block')
    merge_blocks(block_dir)
