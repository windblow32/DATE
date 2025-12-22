import ast
import pickle
import re
from collections import Counter

from ModelShare_with_DSR_final import Rule
from ModelShare_with_DSR_final import Predicate
import numpy as np
import openai
import pandas as pd
from openai import OpenAI
from ModelShare_with_DSR_final import Model

from generate_prompt import promptGenerator, translate_rule
from testF1 import evaluate_singleModel


def parse_generated_data_blocks(generated_text):
    # Initialize an empty list to store DataFrames
    all_dfs = []

    # Regular expression to match lines starting with "Generated Tabular Data"
    pattern = re.compile(r'^.*Generated Tabular Data.*$')

    # Split the text into lines
    lines = generated_text.split('\n')

    # Iterate over the lines
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if the line contains "Generated Tabular Data"
        if pattern.match(line):
            # Check if the next line contains data
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                try:
                    # Parse the data using ast.literal_eval
                    data = ast.literal_eval(data_line)
                    # Convert the data to a DataFrame
                    df = pd.DataFrame(data)
                    all_dfs.append(df)
                    print(f"Successfully parsed {len(df)} rows of data")
                except Exception as e:
                    print(f"Error parsing data block: {str(e)}")
            i += 1  # Skip the data line
        i += 1  # Move to the next line

    return all_dfs
    

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


def decision_path(model, sample, feature_names):
    """获取决策树模型对于一个样本的决策路径"""
    node_indicator = model.decision_path(sample)
    leave_id = model.apply(sample)
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    path_rules = []
    for node_id in node_index:
        if leave_id[0] == node_id:
            break
        else:
            threshold_sign = "<=" if sample[0, feature[node_id]] <= threshold[node_id] else ">"
            feature_name = feature_names[feature[node_id]]
            rule = f"({feature_name} {threshold_sign} {threshold[node_id]})"
            path_rules.append(rule)
    return path_rules


# 设置你的 OpenAI API 密钥
openai.api_key = 'sk-Sz5VQcsOmLGRz0Ne837cEc158d9f477292B856335cEfD361'
# 数据集名称
datasetName = "bank-marketing"
# 数据集路径
json_path = "current_block/bank-marketing_ROU=0.2_f1-error_block.json"

data_path = "dataset/clf_num/" + datasetName + ".csv"
client = OpenAI(api_key=openai.api_key, base_url="https://api.gpt.ge/v1/")

# 存储所有生成的dataframe,最后合并成generated_df,用于模型测试
all_dfs = []
# 存储所有生成的预测结果,用于模型测试
all_pred = []
# 统计生成的规则
path_rules_counter = Counter()
# 存储每个规则对应的所有数据行
rule_to_rows = {}
# 存储历史question
history_question = []
# 历史question中label不一致的数据集合
history_different_rows = []
# 历史question中label一致的数据集合
history_same_rows = []
# 新增：存储每个问题的错误数据与路径配对
question_error_pairs = []  # 每个元素是一个字典，包含question_index和该问题的所有错误配对

# sharedModel id以及data集合
shared_model_data = {}

# 存储example prompt
example_prompt = ""
# 存储每个问题的路径到样本的映射
all_questions_path_samples = []
# 存储表头header
header = []
# 循环18个问题(看current model中的数量)
questionNum = 18
for question_index in range(0, questionNum):
    print(f"\n=== answer question {question_index + 1}/{questionNum} ===")

    # 存储当前问题的错误配对
    current_question_errors = {
        'question_index': question_index,
        'question': None,  # 将在后面填充
        'error_pairs': []  # 存储(data, path)配对
    }

    # 生成提示
    prompt, modelList, header, example_prompt, current_question = promptGenerator(
        current_block_path=json_path,
        dataset_path=data_path,
        question_index=question_index
    )
    history_question.append(current_question)
    current_question_errors['question'] = current_question

    print(prompt)

    # 调用 OpenAI API 进行文本生成
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16000,
            stop=None,
            temperature=0.7
        )

        # 获取并解析生成的文本
        generated_text = response.choices[0].message.content
        # print(f"\n问题 {question_index + 1} 生成的数据：")

        # 解析生成的数据形成DataFrame
        try:
            df = pd.DataFrame(ast.literal_eval(generated_text))
            all_dfs.append(df)
            # print(f"成功解析 {len(df)} 行数据")
        except Exception as e:
            print(f"解析问题 {question_index + 1} 的生成数据时出错: {str(e)}")
            exit(-2)

    except Exception as e:
        print(f"问题 {question_index + 1} 的API调用出错: {str(e)}")
        continue

    # 利用对应的modelID进行测试
    generated_X = df.iloc[:, :-1]
    print(f"生成行数{len(df)}")
    # 检查生成数据
    print(df)
    id = modelList[question_index]
    current_offline_model = load_model(dataset=datasetName, model_id=id)

    m = Model(current_offline_model['original_trainData'])
    m.train()
    pred = m.model.predict(generated_X)
    all_pred.append(pred)

    # 对比预测结果和生成标签
    df_Y = df.iloc[:, -1]
    different_mask = pred != df_Y
    different_rows = df[different_mask]
    history_different_rows.append(different_rows)

    # 预测正确的行
    same_mask = pred == df_Y  # 或者使用 ~different_mask
    same_rows = df[same_mask]
    history_same_rows.append(same_rows)

    # 存储符合要求的id-生成数据集合
    if id not in shared_model_data:
        shared_model_data[id] = same_rows
    else:
        shared_model_data[id] = pd.concat([shared_model_data[id], same_rows], ignore_index=True)

    # 创建修改后的correct_rows，将different_rows的最后一列替换为pred中的值
    correct_rows = different_rows.copy()
    correct_rows.iloc[:, -1] = pred[different_mask]
    print(f"question {question_index + 1} 回答中与pred不同的行数: {len(different_rows)}")

    # 初始化当前问题的路径到样本的映射
    current_path_to_samples = {}
    print(f"modelID: {id}")
    # 对于所有标签不一致的different_rows中的data,分析决策树路径,并统计出现次数最多的路径
    if len(different_rows) > 0:
        decision_tree_model = m.model  # m.model 是一个决策树模型
        feature_names = header  # 获取特征名称列表
        for index, row in different_rows.iterrows():
            sample = row.iloc[:-1].values.reshape(1, -1)
            path_rules = decision_path(decision_tree_model, sample, feature_names)
            path_str = ", ".join(path_rules)
            rule_tuple = tuple(path_rules)

            print(f"Row {index} decision path:")
            print(path_str)

            # 创建数据与路径的配对
            error_pair = {
                'row_index': index,
                'original_data': row.to_dict(),
                'corrected_data': correct_rows.loc[index].to_dict(),
                'path': path_rules,
                'path_string': path_str
            }
            current_question_errors['error_pairs'].append(error_pair)

            # 更新计数器
            path_rules_counter.update([rule_tuple])

            # 将样本添加到当前问题的路径映射中
            if path_str not in current_path_to_samples:
                current_path_to_samples[path_str] = {
                    'rule_tuple': rule_tuple,
                    'samples': []
                }
            current_path_to_samples[path_str]['samples'].append({
                'row_index': index,
                'original_data': row.to_dict(),
                'corrected_data': correct_rows.loc[index].to_dict()
            })

            # 对于label不一致的data，按照shareModel的预测结果修正，并存储修改后的行数据到rule_to_rows
            if rule_tuple not in rule_to_rows:
                rule_to_rows[rule_tuple] = []
            corrected_row = correct_rows.loc[index]
            rule_to_rows[rule_tuple].append(corrected_row)

    # 分析如果把不一致节点加入当前sharedModel中，sharedModel决策树模型的变化情况

    # 得到不一致的路径信息，告诉LLM

    # 保存当前问题的路径到样本的映射
    all_questions_path_samples.append(current_path_to_samples)
    question_error_pairs.append(current_question_errors)

pred_Y = all_pred[0]
if all_pred:
    for array in all_pred[1:]:
        pred_Y = np.concatenate((pred_Y, array), axis=0)

    print(f"已验证{pred_Y.shape[0]}行生成数据")

# 合并所有生成的数据框
if all_dfs:
    generated_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并完成，总共生成 {len(generated_df)} 行数据")

    # 提取特征和标签
    generated_X = generated_df.iloc[:, :-1]
    generated_Y = generated_df.iloc[:, -1]
else:
    print("没有成功生成任何数据")
    exit(1)


# 第一轮，生成新数据，测试效果
print("===========Evaluate===========")
# 用每个shared model去测试error，并最后计算总error
# 每个shared model对应一部分增强的数据
for model_id, df in shared_model_data.items():
    evaluate_singleModel(path=datasetName, additional_data=df.values, model_id=model_id)


# last round question-answer pair, add into Historical Q&A Examples
withhistory_prompt = example_prompt + "### Instructions ###\n"
withhistory_prompt += "You will receive some Historical Q&A Examples as follows. To valid the quality of generated data, we use a decision tree to predict the label for each example. In the answer part of our example, we show the reasons(Decision Tree Path) why the generated data is incorrect. \n"
withhistory_prompt += "You should learn from them, identify which rules may lead to incorrect data, and which rules ensure data correctness. Please pay attention to the following points during your analysis:\n"
withhistory_prompt += f'''
1. **Identify rules for incorrect data**: Analyze the parts of the data that do not have the same label as the predict of Model. Remember the rules that lead to incorrect data, and try to avoid these rules in the subsequent data generation process.
2. **Learn rules for correct data**: Summarize which rules or conditions can ensure data correctness and make sure these rules are applied in the subsequent data generation process.
3. **Notice the scale of data**:In the answer part of our historical examples, the header of records is {header}.
4. **Generate new rules**: You should generate new rules based on the historical examples. Try to learn the relationships described in correct rules, and avoid these incorrect rules in the subsequent data generation process.
5. **Generate tabular data based on new rules**: Based on the analysis above, you should generate tabular data that satisfies the new generated rule.
6. ** Format your answer**: The details of your task and answer are listed at the end of this prompt.\n'''

withhistory_prompt += "### Your Task ###\n"
# 一次生成多条可能的，还是逐条去做
withhistory_prompt += "Give me SOME new rule to generate data for a given model_id=0 that is DIFFERENT from the old ones (but should use the listed attributes)\n"
withhistory_prompt += "The format your answer should be the same as the historical examples. It should be composed of multiple rules like: ['Numerical value of column XXX is greater(lower) than XXX.', 'Numerical value of column XXX is greater(lower) than XXX.', ...]\n"
withhistory_prompt += "After that, generate tabular data in one line for each new rule.\n"

withhistory_prompt += "### Your Answer ###\n"
withhistory_prompt += "For each generated rule, your answer should be:\n"
withhistory_prompt += "Improved Attribute Rule i: \n"
withhistory_prompt += "Generated Tabular Data i: \n"

withhistory_prompt += "### Historical Q&A Examples ###\n"
for i, question in enumerate(history_question):
    withhistory_prompt += f"Example {i + 1}:\n"
    # withhistory_prompt += "### Question ###\n"
    withhistory_prompt += f"Attribute Rule: {question}\n"
    withhistory_prompt += "Extracted from: Model {modelList[i]}\n"
    # withhistory_prompt += "### Answer ###\n"
    withhistory_prompt += f"Correct Generated Tabular Data: \n{header}\n{history_same_rows[i]}\n"
    withhistory_prompt += f"Incorrect Generated Tabular Data: \n"
    # 打印当前问题的路径分组结果
    if not all_questions_path_samples or i >= len(all_questions_path_samples) or not all_questions_path_samples[i]:
        reply = f"No incorrect samples for question {i + 1}\n"
        withhistory_prompt += reply
        print(reply)
    else:
        for path_str, path_data in all_questions_path_samples[i].items():
            samples = path_data['samples']
            for sample in samples:
                reply = f"{list(sample['original_data'].values())} "
                withhistory_prompt += reply
                print(reply, end="")
            print()
            withhistory_prompt += "\n"  # Add a newline after all samples

            reply = f"The labels(the last column) of these samples are wrong. Their prediction path in model is: {path_str}\n"
            print(reply)
            withhistory_prompt += reply


# # 输出统计结果,按照count有高到低排序
# print("\nDecision path statistics:")
# most_common_rules = path_rules_counter.most_common(1)
# if most_common_rules:
#     most_common_rule, count = most_common_rules[0]
#     print(f"Most common rule (count={count}): {most_common_rule}")

#     # 翻译规则
#     translated_rules = [translate_rule(rule) for rule in most_common_rule]
#     print(f"Translated rules: {translated_rules}")

#     # 直接获取之前存储的对应规则的所有行数据
#     matching_rows = rule_to_rows.get(most_common_rule, [])

#     if matching_rows:
#         print(f"Found {len(matching_rows)} rows matching the most common rule")
#         # 之前的prompt是历史记录，把新建的prompt_parts写到后面,并加上模型反馈

#         # 让LLM为我们生成新问题：



# 生成新rule
print("==============RuleGenerator prompt================")
print(withhistory_prompt)

print("==============RuleGenerator answer================")
# 调用 OpenAI API 进行文本生成
generated_text = ""
try:
    client = OpenAI(api_key=openai.api_key, base_url="https://api.v36.cm/v1/")
    response = client.chat.completions.create(
        model="deepseek-r1-250528",
        messages=[{"role": "user", "content": withhistory_prompt}],
        max_tokens=16000,
        stop=None,
        temperature=0.7
    )

    # 获取并解析生成的文本
    generated_text = response.choices[0].message.content
    print(generated_text)
except Exception as e:
    print(f"Error: {e}")


# 解析生成的数据形成DataFrame
generated_dfs = parse_generated_data_blocks(generated_text)

for df in generated_dfs:
    generated_X = df.iloc[:, :-1]
    print(f"生成行数{len(df)}")
    # 检查生成数据
    print(df)
    id = 0
    current_offline_model = load_model(dataset=datasetName, model_id=id)
    m = Model(current_offline_model['original_trainData'])
    m.train()
    pred = m.model.predict(generated_X)

    # 对比预测结果和生成标签
    df_Y = df.iloc[:, -1]

    # 预测正确的行
    same_mask = pred == df_Y  # 或者使用 ~different_mask
    same_rows = df[same_mask]
    history_same_rows.append(same_rows)

    # 存储符合要求的id-生成数据集合，更新生成数据
    if id not in shared_model_data:
        shared_model_data[id] = same_rows
    else:
        shared_model_data[id] = pd.concat([shared_model_data[id], same_rows], ignore_index=True)

    # 测试新rule生成的数据效果能否提升
    evaluate_singleModel(path=datasetName, additional_data=shared_model_data[id].values, model_id=0)




# 对比1: generated row 最后一列替换成pred_Y(说明直接替换，质量也不一定合格)
# generated_df.iloc[:, -1] = pred_Y

# print("share model predict label:")
# print(pred_Y)
# print("Our generated label:")
# print(generated_Y)

# 对比2: Create a new dataframe with rows where the last column matches pred_Y
# filter_df = generated_df[generated_df.iloc[:, -1] == pred_Y]

