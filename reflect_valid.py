import ast
import pickle
import re
import time
from collections import Counter
import numpy as np
import pandas as pd
from openai import OpenAI, APIError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ModelShare_with_DSR_final import Model, Rule, Predicate
from generate_prompt import promptGenerator, translate_rule, stratified_sample
from testF1 import evaluate_singleModel, evaluate_validation


def format_rules(rule_list):
    """
    将形如
      ['Numerical value of column V14 >= 200.0', ...]
    的列表，转换为：
      "(V14 >= 200.0), (V14 <= 350.0), ..."
    """
    formatted = []
    for rule in rule_list:
        # 去掉前缀
        s = rule.replace('Numerical value of column ', '')
        # 去掉多余空白
        s = s.strip()
        # 加上括号
        formatted.append(f'({s})')
    # 用逗号和空格连接
    return ', '.join(formatted)


class ReflectiveDataGenerator:
    def __init__(self, dataset_name, data_path, json_path, api_key, base_url):
        """初始化反思数据生成器"""
        self.initial_questions = -1
        self.init_question_list = []
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.json_path = json_path
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.knowledge_base = []

        # 存储历史信息
        self.all_dfs = []
        self.all_pred = []
        self.path_rules_counter = Counter()
        self.rule_to_rows = {}
        self.history_question = []  # 存储所有历史规则
        self.history_different_rows = []  # 存储所有历史的错误数据
        self.history_same_rows = []  # 存储所有历史的正确数据
        self.history_model_ids = []  # 存储每个历史规则对应的模型ID
        self.question_error_pairs = []
        self.shared_model_data = {}
        self.all_questions_path_samples = []
        self.header = []
        self.example_prompt = ""

        # 存储每个问题的valid_path_to_samples
        self.valid_path_to_samples = {}

        # 轮次信息
        self.current_round = 0
        self.round_history = []

    def save_generated_data(self):

        # 保存shared_model_data中的数据
        for model_id in self.shared_model_data:
            print(f"Saving generated data for model {model_id} ...")
            # 用pickle按照id保存对象
            with open(f"generated/0825_{self.dataset_name}_data_{model_id}.pkl", "wb") as f:
                pickle.dump(self.shared_model_data[model_id], f)
                print(f"Saved model {model_id} to generated/0823_{self.dataset_name}_data_{model_id}.pkl")

    def parse_generated_data_blocks(self, generated_text):
        """
        解析所有 Generated Tabular Data 块，返回一个 DataFrame 列表，
        顺序与它们在原文本中出现的顺序一致。
        既能处理同一行出现 [[…]] 的场景，也能处理跨行的情况。
        """
        dfs = []
        lines = generated_text.split('\n')
        header_pattern = re.compile(r'Generated Tabular Data')

        i = 0
        while i < len(lines):
            line = lines[i]
            # 只要这一行出现关键字，就尝试从该行开始去定位 [[…]] 块
            if header_pattern.search(line):
                # 首先看同一行里有没有 [[…]]
                inline_match = re.search(r'(\[\[.*\]\])', line)
                if inline_match:
                    text_block = inline_match.group(1)
                    end_idx = i + 1
                else:
                    # 否则从下一行开始累积，直到中括号配平
                    text_block = ''
                    bracket_count = 0
                    end_idx = i + 1
                    while end_idx < len(lines):
                        seg = lines[end_idx].strip()
                        bracket_count += seg.count('[') - seg.count(']')
                        text_block += seg
                        end_idx += 1
                        if bracket_count == 0 and text_block.startswith('[['):
                            break

                # 如果拿到了 text_block，就尝试解析
                if text_block.startswith('[[') and text_block.endswith(']]'):
                    try:
                        data = ast.literal_eval(text_block)
                        # 如果是外层一维列表包装一层列表（[[…]] → […])
                        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
                            data = data[0]
                        df = pd.DataFrame(data)
                        dfs.append(df)
                        print(f"Parsed block at lines {i}-{end_idx - 1}, rows={len(df)}")
                    except Exception as e:
                        print(f"Error parsing data block at lines {i}-{end_idx - 1}: {e}")

                # 跳过已消费的行
                i = end_idx
                continue
            i += 1

        return dfs

    def parse_generated_rules(self, text):
        """
        扫描 text 中所有包含 'Improved Attribute Rule' 的行，
        提取方括号 [] 中的内容并解析为 Python 列表，返回一个“列表的列表”。
        """
        pattern = re.compile(r'Improved Attribute Rule[^:]*:\s*(\[[^\]]*\])')
        matches = pattern.findall(text)

        rules = []
        for m in matches:
            try:
                # 真正将字符串 "[...]" 解析成 Python list
                lst = ast.literal_eval(m)
            except Exception:
                # 万一格式不完全合法，再退回手动拆分
                inner = m[1:-1]  # 去掉最外层 [ ]
                parts = [item.strip().strip("'\"") for item in inner.split(',') if item.strip()]
                lst = parts
            rules.append(lst)

        return rules

    def load_model(self, model_id):
        """加载模型"""
        filename = f"model/{self.dataset_name}_model_{model_id}.pkl"
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Model {model_id} not found")
            return None

    def decision_path(self, model, sample, feature_names):
        """获取决策树路径"""
        node_indicator = model.decision_path(sample)
        leave_id = model.apply(sample)
        pred_Y = model.predict(sample)
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
        return path_rules, pred_Y

    def initial_data_generation(self, question_num=18, target_id=2):
        """初始数据生成阶段"""
        print(f"\n=== Round {self.current_round + 1}: Initial Data Generation ===")

        round_data = {
            'round': self.current_round + 1,
            'type': 'initial',
            'questions': [],
            'generated_data': [],
            'evaluation_results': {}
        }
        knowledge_base = []

        for prompt_index in range(question_num):
            print(f"\n--- Question {prompt_index + 1}/{question_num} ---")

            # 生成提示
            prompt, modelList, header, example_prompt, current_question, knowledge_base = promptGenerator(
                current_block_path=self.json_path,
                dataset_path=self.data_path,
                question_index=prompt_index,
                target_id=target_id
            )

            if not self.header:  # 只在第一次设置
                self.header = header
                self.example_prompt = example_prompt

            # 输出prompt size
            print(f"Prompt size: {len(prompt)}")
            # 调用API生成数据
            generated_data = self._call_llm_api(prompt)
            # print(generated_data)
            if generated_data is None:
                continue

            # 解析数据
            try:
                df = pd.DataFrame(ast.literal_eval(generated_data))
                df.columns = header
                self.all_dfs.append(df)
            except Exception as e:
                print(f"Error parsing question index {prompt_index}: {str(e)}")
                continue

            # print(current_question)
            # 验证和分析数据
            analysis_result = self._analyze_generated_data(current_question, df, modelList[prompt_index],
                                                           prompt_index, reflect=False)

            round_data['questions'].append({
                'index': prompt_index,
                'question': current_question,
                'model_id': modelList[prompt_index],
                'generated_df': df,
                'analysis': analysis_result
            })
        # 最后一轮获取knowledge_base，放到history的example中
        self.knowledge_base = knowledge_base
        # 评估当前轮次效果
        evaluation_results = self._evaluate_round()
        round_data['evaluation_results'] = evaluation_results

        self.round_history.append(round_data)
        self.current_round += 1
        return round_data

    def reflective_data_generation(self, model_id):
        """反思数据生成阶段"""
        print(f"\n=== Round {self.current_round + 1}: Reflective Data Generation ===")

        round_data = {
            'round': self.current_round + 1,
            'type': 'reflective',
            'generated_data': [],
            'new_rules': [],
            'evaluation_results': {}
        }

        # 构建反思prompt
        reflective_prompt = self._build_reflective_prompt(target_id=model_id)

        print("==============Reflective Prompt================")
        # print(reflective_prompt)

        # 调用API生成新规则和数据
        generated_text = self._call_llm_api(reflective_prompt, "deepseek-r1-250528", "https://api.v36.cm/v1/")
        if generated_text is None:
            return None

        print("==============Reflective Answer================")
        # print(generated_text)
        print(f"tokens: {len(reflective_prompt) + len(generated_text)}")

        # 解析生成的规则和数据
        generated_rules = self.parse_generated_rules(generated_text)
        generated_dfs = self.parse_generated_data_blocks(generated_text)

        print(f"Parsed {len(generated_rules)} rules and {len(generated_dfs)} data blocks")

        # 统计现在有多少个question
        question_sum = len(self.history_question)

        # 处理每个新生成的规则和数据对
        for i, (rule_List, df) in enumerate(zip(generated_rules, generated_dfs)):
            print(f"\n--- Processing Rule-Data Pair {i + 1} ---")
            rule = format_rules(rule_List)
            # print(f"Rule: {rule}")
            # print(f"Data shape: {df.shape}")

            question_sum = question_sum + i + 1
            # 分析新生成的数据
            analysis_result = self._analyze_generated_data(rule, df, model_id, question_sum, reflect=True)

            round_data['new_rules'].append(rule)
            round_data['generated_data'].append({
                'rule': rule,
                'generated_df': df,
                'analysis': analysis_result
            })
            # 对于correct_rows，进行模型验证
            if model_id not in self.shared_model_data or not self.shared_model_data[model_id]:
                additional_data = None
            else:
                # 从shared_model_data中提取所有data_list并合并成一个列表
                additional_data = []
                for data_obj in self.shared_model_data[model_id]:
                    additional_data.extend(data_obj['data_list'])

            evaluate_singleModel(path=self.dataset_name, additional_data=additional_data,
                                 model_id=model_id)

        # 评估当前轮次效果
        evaluation_results = self._evaluate_round()
        round_data['evaluation_results'] = evaluation_results

        self.round_history.append(round_data)
        self.current_round += 1
        return round_data

    def _add_to_history(self, rule, data, model_id, score):
        """将新规则和数据加入到历史记录中"""
        print(f"Adding new rule to history: {rule}")
        # 新的question_index，注意加在history_question.append(rule)之前。（例如原来question数量为1，index=0，那么现在就应该index=1=num）
        question_index = str(len(self.history_question))

        # 将新规则和数据加入history, modelID等
        self.history_question.append(rule)
        self.history_model_ids.append(model_id)

        if question_index not in self.valid_path_to_samples:
            self.valid_path_to_samples[question_index] = {}
        # 存储路径信息
        if rule not in self.valid_path_to_samples[question_index]:
            self.valid_path_to_samples[question_index][rule] = {
                'rule_tuple': rule,
                'samples': data,
                'modelID': model_id,
                'score': score,
                'pred_Y': None
            }
        else:
            print("rule already in history")

    def _call_llm_api(self, prompt, model="deepseek-r1-250528", base_url="https://api.gpt.ge/v1/", retries=3, delay=2):
        for attempt in range(1, retries + 1):
            try:
                if base_url != "https://api.gpt.ge/v1/":
                    client = OpenAI(api_key=self.client.api_key, base_url=base_url)
                else:
                    client = self.client
                # 16000 response
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32000,
                    stop=None,
                    temperature=0.7
                )
                return response.choices[0].message.content
            except APIError as e:
                if attempt < retries:
                    print(f"502 错误，{delay} 秒后重试 ({attempt}/{retries})...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    raise

    def _analyze_generated_data(self, rule, df, model_id, prompt_index, reflect):
        '''

        :param rule: 解析出的规则
        :param df: 保存的数据
        :param model_id: 当前处理的模型id
        :param prompt_index: prompt中question的index
        :param reflect: boolean, 表示这个分析过程是否存在于reflect过程。只有reflect的结果，最后才有用
        :return:
        '''
        # 这里的prompt index是生成数据时候的question
        df.columns = self.header
        # print(df)
        """分析生成的数据"""
        # 对于init的部分，history_question还没有添加,这个时候index就是len
        question_index = str(len(self.history_question))
        self.init_question_list.append(question_index)
        # 一个question，answer分析很多
        self.history_question.append(rule)
        self.history_model_ids.append(model_id)
        # print(f"Analyzing {len(df)} rows of data for model {model_id}, question_key: {prompt_index}")

        # 加载模型
        current_offline_model = self.load_model(model_id)
        if current_offline_model is None:
            return None

        m = Model(current_offline_model['final_trainData'])
        m.train()

        decision_tree_model = m.model

        # 分析内部成分
        # 为生成的df计算决策路径，并按照不同路径划分成子集
        # 确保valid_path_to_samples已初始化
        if not hasattr(self, 'valid_path_to_samples'):
            self.valid_path_to_samples = {}

        # 确保当前question_key的字典已初始化
        if question_index not in self.valid_path_to_samples:
            self.valid_path_to_samples[question_index] = {}

        for index, row in df.iterrows():
            sample = row.iloc[:-1].values.reshape(1, -1)
            path_rules, pred_Y = self.decision_path(decision_tree_model, sample, self.header)
            path_str = ", ".join(path_rules)
            rule_tuple = tuple(path_rules)
            # 某些rule是空的
            if path_str == "" or path_str is None:
                path_str = "EMPTY"

            # 存储路径信息
            if path_str not in self.valid_path_to_samples[question_index]:
                self.valid_path_to_samples[question_index][path_str] = {
                    'rule_tuple': rule_tuple,
                    'samples': [],
                    'modelID': model_id,
                    'score': 0,
                    'pred_Y': pred_Y
                }
            self.valid_path_to_samples[question_index][path_str]['samples'].append(row.to_dict())

        # valid_path_to_samples中的每个路径对应的samples进行模型验证，存储为pair
        for path_str, pair in self.valid_path_to_samples[question_index].items():
            # 提取所有original_data
            data_list = []
            for sample in pair['samples']:
                data_list.append(list(sample.values()))

            data_array = np.array(data_list)
            # 计算error rate改进了多少
            valid_gain = evaluate_validation(path=self.dataset_name, additional_data=data_array,
                                             model_id=model_id)
            self.valid_path_to_samples[question_index][path_str]['score'] = valid_gain
            # 所有正反馈数据加入到shared_model_data，顺便检查下evaluate_round
            if valid_gain > 0.0:
                # print("*******new rule********")
                # print(f"path: {path_str}")
                # print(valid_gain)

                # 创建包含data_list, path和score的对象
                data_object = {
                    'data_list': data_array.tolist(),  # 转换为Python原生list
                    'path': path_str,
                    'score': float(valid_gain),  # 确保是Python原生float
                    'reflect': reflect
                }

                # 初始化model_id的列表（如果不存在）
                if model_id not in self.shared_model_data:
                    self.shared_model_data[model_id] = []

                # 添加新的数据对象
                self.shared_model_data[model_id].append(data_object)
            else:
                # valid降低，反思路径问题
                # 对比path_str与rule之间的冲突
                print(f"not valid gain: {valid_gain}, decision tree path: {path_str}")

            # 将新规则和数据加入历史记录中
            self._add_to_history(path_str, pair['samples'], model_id, valid_gain)

        # 直接作为整体加入model中查看validation score
        total_valid_gain = evaluate_validation(path=self.dataset_name, additional_data=df,
                                               model_id=model_id)
        print(f"Total valid gain: {total_valid_gain}")
        records = df.to_dict(orient='records')
        self._add_to_history(rule, records, model_id, total_valid_gain)

        # 预测
        generated_X = df.iloc[:, :-1]
        pred = m.model.predict(generated_X)
        self.all_pred.append(pred)

        # 对比预测结果和生成标签
        df_Y = df.iloc[:, -1]
        different_mask = pred != df_Y
        different_rows = df[different_mask]
        same_mask = pred == df_Y  # 或者使用 ~different_mask
        same_rows = df[same_mask]
        # 存储结果（只有初始阶段才添加到历史记录）
        self.history_different_rows.append(different_rows)
        self.history_same_rows.append(same_rows)

        # 分析决策路径（只有初始阶段才添加到历史记录）
        different_path_to_samples = {}
        if len(different_rows) > 0:
            correct_rows = different_rows.copy()
            correct_rows.iloc[:, -1] = pred[different_mask]

            decision_tree_model = m.model
            for index, row in different_rows.iterrows():
                sample = row.iloc[:-1].values.reshape(1, -1)
                path_rules, pred_Y = self.decision_path(decision_tree_model, sample, self.header)
                path_str = ", ".join(path_rules)
                rule_tuple = tuple(path_rules)

                # 更新统计
                self.path_rules_counter.update([rule_tuple])

                # 存储路径信息
                if path_str not in different_path_to_samples:
                    different_path_to_samples[path_str] = {
                        'rule_tuple': rule_tuple,
                        'samples': []
                    }
                different_path_to_samples[path_str]['samples'].append({
                    'row_index': index,
                    'original_data': row.to_dict(),
                    'corrected_data': correct_rows.loc[index].to_dict()
                })

            # 存储需要处理的路径
            paths_to_remove = []
            paths_data_to_add = []

            # 不一致数据放到model中验证
            # print("Analyzing different row with rules")
            for path_str, pair in different_path_to_samples.items():
                # 提取所有original_data
                original_data_list = []

                for sample in pair['samples']:
                    original_data_list.append(list(sample['original_data'].values()))

                # 转换为numpy数组或DataFrame进行评估
                original_data_array = np.array(original_data_list)
                # 输出模型的验证集表现的提升程度
                result = evaluate_validation(path=self.dataset_name, additional_data=original_data_array,
                                             model_id=model_id)

                if result > 0:
                    # 不一致的数据对于model有贡献，对应的规则应该保留
                    # print(f"不一致的数据对于model有贡献，路径: {path_str}")
                    paths_to_remove.append(path_str)

                    # 准备要添加的数据
                    sample_data = []
                    sample_indices = []
                    for sample in pair['samples']:
                        sample_data.append(list(sample['original_data'].values()))
                        sample_indices.append(sample['row_index'])

                    paths_data_to_add.append({
                        'data': pd.DataFrame(sample_data, columns=self.header),
                        'indices': sample_indices
                    })
                else:
                    # 对应的规则彻底是inconsistent的，最后作为负面例子
                    print(f"对应的规则彻底是inconsistent的，路径: {path_str}")

            # 处理需要移动的数据
            for i, path_str in enumerate(paths_to_remove):
                # 从different_path_to_samples中删除
                different_path_to_samples.pop(path_str)

                # 添加到shared_model_data
                data_to_add = paths_data_to_add[i]['data']
                data_to_add.columns = self.header
                # 添加到same_rows
                same_rows = pd.concat([same_rows, data_to_add], ignore_index=True)

                # 从different_rows中删除对应的行
                indices_to_drop = paths_data_to_add[i]['indices']
                different_rows = different_rows.drop(indices_to_drop)

        self.all_questions_path_samples.append(different_path_to_samples)

        return {
            'correct_rows': len(same_rows),
            'incorrect_rows': len(different_rows),
            'accuracy': len(same_rows) / len(df) if len(df) > 0 else 0,
            'path_samples': different_path_to_samples,
            'total_valid_gain': total_valid_gain
        }, same_rows

    def _build_reflective_prompt(self, target_id):
        '''
        把question中问不完的部分，划分到知识库，并且按照相同的格式给出validScore
        :param target_id:
        :return:
        '''
        classLabel1 = 0
        classLabel2 = 1
        """构建反思prompt，包含所有历史example和模型数据"""
        # 加载 model_id=2 的模型数据
        model_data = self.load_model(model_id=target_id)
        if model_data is None:
            print("Model ID 2 not found or failed to load.")
            return None

        # 提取模型数据中的关键信息
        original_train_data = model_data.get('original_trainData', [])
        class_distribution = Counter([row[-1] for row in original_train_data])
        total_samples = len(original_train_data)
        class_1_count = class_distribution.get(classLabel1, 0)
        class_2_count = class_distribution.get(classLabel2, 0)

        # 获取模型原始的base_score（accuracy）
        # model_data['final_trainData']划分为验证集合
        data_v = pd.DataFrame(original_train_data).copy()
        X_train, X_valid, y_train, y_valid = train_test_split(data_v.iloc[:, :-1], data_v.iloc[:, -1], test_size=0.2,
                                                              random_state=42)
        # X_train, y_train合并为输入数据trainData
        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_valid)
        base_score = accuracy_score(y_valid, y_pred)

        """构建反思prompt，包含所有历史example"""
        withhistory_prompt = self.example_prompt + "\n### Instructions ###\n"
        # withhistory_prompt = "### Instructions ###\n"
        withhistory_prompt += f"You will receive some Historical Q&A Examples as follows. Each example includes a question rule, generated data, the decision tree path, and the validation score. The question rule guides the generation of data. To valid the quality of generated data, we use the decision tree to reason the path for each example. In the answer part of our example, we extract the prediction Path(which can be translated into rule) in the Decision Tree for each generated data. To evaluate the path(rule), we collect the data that share the same Path, and add them to the training data of the model. Finally, we get the increase in accuracy as the validation score."
        withhistory_prompt += "In Historical Q&A Examples, generated data may have different label from the predict of Decision Tree. These part of data makes our model more ROBUST, and can still increase the accuracy of model.\n"
        withhistory_prompt += "You should learn to generate high validation score rules, and generate tabular data based on new rules and above analysis. Please pay attention to the following points during your analysis:\n"
        withhistory_prompt += f'''
1. **Focus on target model**: Each question is focus on a specific model_id. You should focus on the target model_id={target_id}. When generating rules and data, you should always prioritize the current model's preferences as the primary framework. {f"To generate more data with label={classLabel1}, you can learn the knowledge from Model 0" if class_1_count < class_2_count else ""}
2. **Balance the class while generating**: To avoid overfit, you should generate more records with class={classLabel1 if class_1_count < class_2_count else classLabel2}. The current training data is composed of {class_2_count} records with class={classLabel2}, and {class_1_count} records with class={classLabel1}. 
3. **Learn the features of high score rules**: You should learn from rules with high validation score from the historical examples. Compare new rules with the question rule and think how to build a new rule than can maximize the validation score. Rethink what details in the additional part of some rules can improve the performance of target model. 
4. **Avoid the mistake from low score rules**: Analyze the difference between these rules with low validation score and the original rule in the question. Analyze which changes of attributes in rules have led to a decline in the model's performance.
5. **Notice the format of data**:In the answer part of our historical examples, the header of records is {self.header}.
6. **Generate new rules**: You should generate new rules based on the historical examples. The generated rules should be as detailed as possible and encompass your knowledge regarding the enhancement of the valid score. For example, the format of a rule should be: ['Numerical value of column V14 < 355', 'Numerical value of column V12 < 600'], where V14 and V12 are attributes. Impose constraints on multiple attribute values to generate data that meets the criteria for a high validation score.
7. **Generate tabular data based on new rules**: Based on the analysis above, you should generate tabular data that satisfies the new generated rule. For example: [[],[],..,[]]
8. **Format your answer**: The details of your task and answer are listed at the end of this prompt.'''

        withhistory_prompt += "### Your Task ###\n"
        withhistory_prompt += f"Give me SOME new rule to generate data for the given model_id={target_id}\n"
        withhistory_prompt += "The format your answer should be the same as the historical examples. It should be composed of multiple rules like: ['Numerical value of column XXX >(or <) XXX.', 'Numerical value of column XXX >(or <) XXX.', ...]\n"
        withhistory_prompt += "After that, generate tabular data in one line for each new rule.\n"

        withhistory_prompt += "### Your Answer ###\n"
        withhistory_prompt += "For each generated rule, your answer should be:\n"
        withhistory_prompt += "Improved Attribute Rule i: \n"
        withhistory_prompt += "Generated Tabular Data i: \n"

        withhistory_prompt += "### Historical Q&A Examples ###\n"
        # format
        # knowledge_base.append({
        #     'rule': str(rule_key),
        #     'support': support,
        #     'modelID': modelID,
        #     'data': data_list,
        #     'validScore': validScore
        # })
        sampleSize = 30
        # 先放入知识库
        for j, knowledge in enumerate(self.knowledge_base):
            withhistory_prompt += f"Example {j}:\n"
            withhistory_prompt += f"(1)Question Rule: {knowledge['rule']}\n"
            withhistory_prompt += f"(2)Focus on: Model {knowledge['modelID']}\n"
            withhistory_prompt += f"(3)Answer:  Generated Data\n"
            # 展示最多30条
            withhistory_prompt += f"Generated Data: {stratified_sample(knowledge['data'], sampleSize) if len(knowledge['data']) >= sampleSize else knowledge['data']}\n"
            withhistory_prompt += f"Added into training data, validation score change: {knowledge['validScore'] - base_score}\n"

        knowledge_size = len(self.knowledge_base)

        # 包含所有历史example（初始的 + 反思生成的）
        for i, question in enumerate(self.history_question):
            # 与当前model无关的example不要
            if self.history_model_ids[i] != 0 and self.history_model_ids[i] != target_id:
                continue

            # 对于输出一个question下子问题
            if str(i) in self.init_question_list:
                continue
            withhistory_prompt += f"Example {i + knowledge_size}:\n"
            withhistory_prompt += f"(1)Question Rule: {question}\n"

            # 添加模型ID信息
            if i < len(self.history_model_ids):
                withhistory_prompt += f"(2)Focus on: Model {self.history_model_ids[i]}\n"

            # 输出所有valid_path_to_samples中的path，data
            if i < len(self.valid_path_to_samples) and self.valid_path_to_samples:
                withhistory_prompt += f"(3)Answer: Generated Data and Decision Tree Path: \n"
                # 第i个question的valid_path_to_samples
                for path_str, path_data in self.valid_path_to_samples[str(i)].items():
                    if path_str == "EMPTY":
                        continue
                    samples = path_data['samples']
                    withhistory_prompt += f"Decision Tree Path: {path_str}\n"
                    if path_data['pred_Y'] is not None:
                        withhistory_prompt += f"Predicted Label: {path_data['pred_Y']}\n"
                    withhistory_prompt += "Generated Data: "
                    for sample in samples:
                        # withhistory_prompt += f"{list(sample.values())} "
                        withhistory_prompt += f"Generated Data: {stratified_sample(list(sample.values()), sampleSize) if len(list(sample.values())) >= sampleSize else list(sample.values())}\n"

                    withhistory_prompt += "\n"
                    withhistory_prompt += f"Added into training data, validation score change: {self.valid_path_to_samples[str(i)][path_str]['score']}\n"

        return withhistory_prompt

    def _evaluate_round(self):
        """评估当前轮次的效果"""
        print("===========Evaluating Round===========")
        evaluation_results = {}
        # shared_model_data应该存储data,score,path,s
        for model_id in self.shared_model_data:
            print(f"Evaluating model {model_id} ...")
            # results = evaluate_greedy_model(path=self.dataset_name, model_id=model_id, additional_data=self.shared_model_data[model_id])
            # evaluation_results[model_id] = results
            df = pd.DataFrame()
            for obj in self.shared_model_data[model_id]:
                data = obj['data_list']
                df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

            if len(df) > 0:
                result = evaluate_singleModel(path=self.dataset_name, additional_data=df.values, model_id=model_id)
                evaluation_results[model_id] = result
                print(f"Model {model_id}: {len(df)} samples")
            else:
                print(f"No generated data for model {model_id} in shared_model_data")

        # 保存self.shared_model_data中的数据，便于查看
        self.save_generated_data()

        return evaluation_results

    def run_multi_round_generation(self, num_rounds=3, initial_questions=3, target_id=2):
        self.initial_questions = 3
        """运行多轮反思生成"""
        print(f"Starting {num_rounds} rounds of reflective data generation")

        # 第一轮：初始数据生成
        initial_result = self.initial_data_generation(initial_questions, target_id)
        # 后续轮次：反思数据生成
        for round_num in range(1, num_rounds):
            startTime = time.time()

            print(f"\n{'=' * 60}")
            print(f"Starting Round {round_num + 1} of {num_rounds}")
            print(f"Current history examples: {len(self.history_question)}")
            print(f"{'=' * 60}")

            reflective_result = self.reflective_data_generation(model_id=target_id)
            if reflective_result is None:
                print(f"Round {round_num + 1} failed, stopping.")
                break

            # fixme 历史example变量更新
            print(f"After round {round_num + 1}, total history examples: {len(self.history_question)}")

            # 可以在这里添加早停条件
            # if self._should_stop():
            #     break
            nowTime = time.time()
            print(f"running time: {nowTime - startTime}")
        return self.get_summary()

    def get_summary(self):
        """获取总结信息"""
        summ = {
            'total_rounds': self.current_round,
            'total_data_generated': sum(len(df) for df in self.all_dfs),
            'total_history_examples': len(self.history_question),
            'shared_model_data': {k: len(v) for k, v in self.shared_model_data.items()},
            'round_history': self.round_history,
            'final_evaluation': self._evaluate_round()
        }

        return summ

    def checkScale(self, model_id):
        if self.shared_model_data[model_id].shape[1] != 8:
            return False
        else:
            return True


# 使用示例
def main():
    # 配置参数
    dataset_name = "credit"
    data_path = "dataset/clf_num/" + dataset_name + ".csv"
    json_path = "current_block/" + dataset_name + "_ROU=0.2_f1-error_block.json"
    api_key = 'sk-Sz5VQcsOmLGRz0Ne837cEc158d9f477292B856335cEfD361'
    base_url = "https://api.gpt.ge/v1/"

    # 创建生成器
    generator = ReflectiveDataGenerator(
        dataset_name=dataset_name,
        data_path=data_path,
        json_path=json_path,
        api_key=api_key,
        base_url=base_url
    )
    # subset分布{0: 72, 1: 123, 2: 34, 3: 64, 4: 80, 5: 23}

    # 运行多轮生成
    summary1 = generator.run_multi_round_generation(num_rounds=20, initial_questions=2, target_id=1)
    summary3 = generator.run_multi_round_generation(num_rounds=20, initial_questions=1, target_id=0)
    # summary6 = generator.run_multi_round_generation(num_rounds=9, initial_questions=95, target_id=5)
    # # summary2 = generator.run_multi_round_generation(num_rounds=9, initial_questions=195, target_id=1)
    # summary4 = generator.run_multi_round_generation(num_rounds=9, initial_questions=136, target_id=3)
    # summary5 = generator.run_multi_round_generation(num_rounds=9, initial_questions=152, target_id=4)

    return summary3


if __name__ == "__main__":
    main()
