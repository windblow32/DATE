import ast
import pickle
import re
from collections import Counter
import numpy as np
import pandas as pd
from openai import OpenAI
from ModelShare_with_DSR_final import Model, Rule, Predicate
from generate_prompt import promptGenerator, translate_rule
from testF1 import evaluate_singleModel


class ReflectiveDataGenerator:
    def __init__(self, dataset_name, data_path, json_path, api_key, base_url):
        """初始化反思数据生成器"""
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.json_path = json_path
        self.client = OpenAI(api_key=api_key, base_url=base_url)

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

        # 轮次信息
        self.current_round = 0
        self.round_history = []

    def parse_generated_data_blocks(self, generated_text):
        """解析生成的数据块"""
        all_dfs = []
        pattern = re.compile(r'^.*Generated Tabular Data.*$')
        lines = generated_text.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if pattern.match(line):
                if i + 1 < len(lines):
                    data_line = lines[i + 1].strip()
                    try:
                        data = ast.literal_eval(data_line)
                        df = pd.DataFrame(data)
                        all_dfs.append(df)
                        print(f"Successfully parsed {len(df)} rows of data")
                    except Exception as e:
                        print(f"Error parsing data block: {str(e)}")
                i += 1
            i += 1
        return all_dfs

    def parse_generated_rules(self, generated_text):
        """解析生成的规则"""
        rules = []
        lines = generated_text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            # 查找规则行，通常以 "Improved Attribute Rule" 开头
            if re.match(r'^.*Improved Attribute Rule \d+.*$', line):
                # 查找下一行的规则内容
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1  # 跳过空行

                if j < len(lines):
                    rule_line = lines[j].strip()
                    try:
                        # 尝试解析规则
                        if rule_line.startswith('[') and rule_line.endswith(']'):
                            rule = ast.literal_eval(rule_line)
                            rules.append(rule)
                        elif rule_line.startswith('`[') and rule_line.endswith(']`'):
                            # 处理markdown格式的规则
                            rule_content = rule_line[1:-1]  # 去掉前后的`
                            rule = ast.literal_eval(rule_content)
                            rules.append(rule)
                    except Exception as e:
                        print(f"Error parsing rule: {rule_line}, Error: {e}")

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

    def initial_data_generation(self, question_num=18):
        """初始数据生成阶段"""
        print(f"\n=== Round {self.current_round + 1}: Initial Data Generation ===")

        round_data = {
            'round': self.current_round + 1,
            'type': 'initial',
            'questions': [],
            'generated_data': [],
            'evaluation_results': {}
        }

        for question_index in range(question_num):
            print(f"\n--- Question {question_index + 1}/{question_num} ---")

            # 生成提示
            prompt, modelList, header, example_prompt, current_question = promptGenerator(
                current_block_path=self.json_path,
                dataset_path=self.data_path,
                question_index=question_index
            )

            if not self.header:  # 只在第一次设置
                self.header = header
                self.example_prompt = example_prompt

            self.history_question.append(current_question)
            self.history_model_ids.append(modelList[question_index])

            # 调用API生成数据
            generated_data = self._call_llm_api(prompt, "gpt-3.5-turbo-16k")
            if generated_data is None:
                continue

            # 解析数据
            try:
                df = pd.DataFrame(ast.literal_eval(generated_data))
                df.columns = header
                self.all_dfs.append(df)
            except Exception as e:
                print(f"Error parsing question {question_index + 1}: {str(e)}")
                continue

            # 验证和分析数据
            analysis_result = self._analyze_generated_data(df, modelList[question_index], question_index)

            round_data['questions'].append({
                'index': question_index,
                'question': current_question,
                'model_id': modelList[question_index],
                'generated_df': df,
                'analysis': analysis_result
            })

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
        print(reflective_prompt[:1000] + "..." if len(reflective_prompt) > 1000 else reflective_prompt)

        # 调用API生成新规则和数据
        generated_text = self._call_llm_api(reflective_prompt, "deepseek-r1-250528", "https://api.v36.cm/v1/")
        if generated_text is None:
            return None

        print("==============Reflective Answer================")
        print(generated_text[:1000] + "..." if len(generated_text) > 1000 else generated_text)

        # 解析生成的规则和数据
        generated_rules = self.parse_generated_rules(generated_text)
        generated_dfs = self.parse_generated_data_blocks(generated_text)

        print(f"Parsed {len(generated_rules)} rules and {len(generated_dfs)} data blocks")

        # 处理每个新生成的规则和数据对
        for i, (rule, df) in enumerate(zip(generated_rules, generated_dfs)):
            print(f"\n--- Processing Rule-Data Pair {i + 1} ---")
            print(f"Rule: {rule}")
            print(f"Data shape: {df.shape}")

            # 分析新生成的数据
            analysis_result = self._analyze_generated_data(df, model_id=model_id, question_index=-1)

            # 将新规则和数据加入历史记录中
            self._add_to_history(rule, df, analysis_result, model_id=0)

            round_data['new_rules'].append(rule)
            round_data['generated_data'].append({
                'rule': rule,
                'generated_df': df,
                'analysis': analysis_result
            })
            # 对于correct_rows，进行模型验证
            evaluate_singleModel(path=self.dataset_name, additional_data=self.shared_model_data[model_id].values,
                                 model_id=model_id)

        # 评估当前轮次效果
        evaluation_results = self._evaluate_round()
        round_data['evaluation_results'] = evaluation_results

        self.round_history.append(round_data)
        self.current_round += 1
        return round_data

    def _add_to_history(self, rule, df, analysis_result, model_id):
        """将新规则和数据加入到历史记录中"""
        print(f"Adding new rule to history: {rule}")

        # 加载模型进行预测
        current_offline_model = self.load_model(model_id)
        if current_offline_model is None:
            return

        m = Model(current_offline_model['final_trainData'])
        m.train()

        # 预测
        generated_X = df.iloc[:, :-1]
        pred = m.model.predict(generated_X)

        # 对比预测结果和生成标签
        df_Y = df.iloc[:, -1]
        different_mask = pred != df_Y
        different_rows = df[different_mask]
        same_rows = df[pred == df_Y]

        # 将新规则和数据加入历史记录
        self.history_question.append(rule)
        self.history_model_ids.append(model_id)
        self.history_different_rows.append(different_rows)
        self.history_same_rows.append(same_rows)

        # 分析决策路径（对错误数据）
        current_path_to_samples = {}
        if len(different_rows) > 0:
            correct_rows = different_rows.copy()
            correct_rows.iloc[:, -1] = pred[different_mask]

            decision_tree_model = m.model
            for index, row in different_rows.iterrows():
                sample = row.iloc[:-1].values.reshape(1, -1)
                path_rules = self.decision_path(decision_tree_model, sample, self.header)
                path_str = ", ".join(path_rules)
                rule_tuple = tuple(path_rules)

                # 更新统计
                self.path_rules_counter.update([rule_tuple])

                # 存储路径信息
                if path_str not in current_path_to_samples:
                    current_path_to_samples[path_str] = {
                        'rule_tuple': rule_tuple,
                        'modelID': model_id,
                        'samples': []
                    }
                current_path_to_samples[path_str]['samples'].append({
                    'row_index': index,
                    'original_data': row.to_dict(),
                    'corrected_data': correct_rows.loc[index].to_dict()
                })

        self.all_questions_path_samples.append(current_path_to_samples)

        print(f"Added to history: {len(same_rows)} correct, {len(different_rows)} incorrect samples")

    def _call_llm_api(self, prompt, model="deepseek-r1-250528", base_url="https://api.gpt.ge/v1/"):
        """调用LLM API"""
        try:
            if base_url != "https://api.gpt.ge/v1/":
                client = OpenAI(api_key=self.client.api_key, base_url=base_url)
            else:
                client = self.client

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16000,
                stop=None,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call error: {str(e)}")
            return None

    def _analyze_generated_data(self, df, model_id, question_index):
        df.columns = self.header
        print(df)
        """分析生成的数据"""
        print(f"Analyzing {len(df)} rows of data for model {model_id}")

        # 加载模型
        current_offline_model = self.load_model(model_id)
        if current_offline_model is None:
            return None

        m = Model(current_offline_model['final_trainData'])
        m.train()

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
        if question_index >= 0:
            self.history_different_rows.append(different_rows)
            self.history_same_rows.append(same_rows)

        # 更新共享模型数据（所有阶段都更新）
        if model_id not in self.shared_model_data:
            self.shared_model_data[model_id] = same_rows
        else:
            self.shared_model_data[model_id] = pd.concat([self.shared_model_data[model_id], same_rows],
                                                         ignore_index=True)

        if not self.checkScale(model_id):
            exit(0)

        # 分析决策路径（只有初始阶段才添加到历史记录）
        current_path_to_samples = {}
        if len(different_rows) > 0:
            correct_rows = different_rows.copy()
            correct_rows.iloc[:, -1] = pred[different_mask]

            decision_tree_model = m.model
            for index, row in different_rows.iterrows():
                sample = row.iloc[:-1].values.reshape(1, -1)
                path_rules = self.decision_path(decision_tree_model, sample, self.header)
                path_str = ", ".join(path_rules)
                rule_tuple = tuple(path_rules)

                # 更新统计
                self.path_rules_counter.update([rule_tuple])

                # 存储路径信息
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

            # 存储需要处理的路径
            paths_to_remove = []
            paths_data_to_add = []

            # 不一致数据放到model中验证
            print("Analyzing different row with rules")
            for path_str, pair in current_path_to_samples.items():
                # 提取所有original_data
                original_data_list = []

                for sample in pair['samples']:
                    original_data_list.append(list(sample['original_data'].values()))

                # 转换为numpy数组或DataFrame进行评估
                original_data_array = np.array(original_data_list)

                result = evaluate_singleModel(path=self.dataset_name, additional_data=original_data_array,
                                              model_id=model_id)

                if result >= 0:
                    # 不一致的数据对于model有贡献，对应的规则应该保留
                    print(f"不一致的数据对于model有贡献，路径: {path_str}")
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
                # 从current_path_to_samples中删除
                current_path_to_samples.pop(path_str)

                # 添加到shared_model_data
                data_to_add = paths_data_to_add[i]['data']
                data_to_add.columns = self.header
                # 添加到same_rows
                same_rows = pd.concat([same_rows, data_to_add], ignore_index=True)
                # 添加到share
                if model_id not in self.shared_model_data:
                    self.shared_model_data[model_id] = same_rows
                else:
                    self.shared_model_data[model_id].columns = self.header
                    self.shared_model_data[model_id] = pd.concat([self.shared_model_data[model_id], data_to_add],
                                                                 ignore_index=True)
                if not self.checkScale(model_id):
                    exit(0)
                # 从different_rows中删除对应的行
                indices_to_drop = paths_data_to_add[i]['indices']
                different_rows = different_rows.drop(indices_to_drop)

        if question_index >= 0:  # 只有初始阶段才添加到历史记录
            self.all_questions_path_samples.append(current_path_to_samples)

        return {
            'correct_rows': len(same_rows),
            'incorrect_rows': len(different_rows),
            'accuracy': len(same_rows) / len(df) if len(df) > 0 else 0,
            'path_samples': current_path_to_samples
        }, same_rows

    def _build_reflective_prompt(self, target_id):
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
        class_1_count = class_distribution.get(1, 0)
        class_2_count = class_distribution.get(2, 0)

        """构建反思prompt，包含所有历史example"""
        withhistory_prompt = self.example_prompt + "### Instructions ###\n"
        withhistory_prompt += "You will receive some Historical Q&A Examples as follows. To valid the quality of generated data, we use a decision tree to predict the label for each example. In the answer part of our example, we show the reasons(Decision Tree Path) why the generated data is incorrect. \n"
        withhistory_prompt += "You should learn from them, identify which rules may lead to incorrect data, and which rules ensure data correctness. Please pay attention to the following points during your analysis:\n"
        withhistory_prompt += f'''
1. **Identify rules for incorrect data**: Analyze the parts of the data that do not have the same label as the predict of Model. Remember the rules that lead to incorrect data, and try to avoid these rules in the subsequent data generation process.
2. **Learn rules for correct data**: Summarize which rules or conditions can ensure data correctness and make sure these rules are applied in the subsequent data generation process. 
3. **Notice the scale of data**:In the answer part of our historical examples, the header of records is {self.header}.
4. **Generate new rules**: You should generate new rules based on the historical examples. Try to learn the relationships described in correct rules, and avoid these incorrect rules in the subsequent data generation process.
5. **Generate tabular data based on new rules**: Based on the analysis above, you should generate tabular data that satisfies the new generated rule.
6. **Focus on target model**: Each question is focus on a specific model_id. When generating rules and data, you should always prioritize the current model's preferences as the primary framework, then leverage knowledge from other models to supplement your reasoning.
7. **Balance the class while generating**: The current training data is composed of {class_2_count} records with class=2, and {class_1_count} records with class=1. To avoid overfit, you should generate more records with class={1 if class_1_count < class_2_count else 2}.
8. **Format your answer**: The details of your task and answer are listed at the end of this prompt.\n'''

        withhistory_prompt += "### Your Task ###\n"
        withhistory_prompt += f"Give me SOME new rule to generate data for a given model_id={target_id} that is DIFFERENT from the old ones (but should use the listed attributes)\n"
        withhistory_prompt += "The format your answer should be the same as the historical examples. It should be composed of multiple rules like: ['Numerical value of column XXX is greater(lower) than XXX.', 'Numerical value of column XXX is greater(lower) than XXX.', ...]\n"
        withhistory_prompt += "After that, generate tabular data in one line for each new rule.\n"

        withhistory_prompt += "### Your Answer ###\n"
        withhistory_prompt += "For each generated rule, your answer should be:\n"
        withhistory_prompt += "Improved Attribute Rule i: \n"
        withhistory_prompt += "Generated Tabular Data i: \n"

        withhistory_prompt += "### Historical Q&A Examples ###\n"

        # 包含所有历史example（初始的 + 反思生成的）
        for i, question in enumerate(self.history_question):
            withhistory_prompt += f"Example {i + 1}:\n"
            withhistory_prompt += f"Attribute Rule: {question}\n"

            # 添加模型ID信息
            if i < len(self.history_model_ids):
                withhistory_prompt += f"Focus on: Model {self.history_model_ids[i]}\n"

            # 添加正确数据
            if i < len(self.history_same_rows):
                withhistory_prompt += f"Correct Generated Tabular Data: \n{self.header}\n{self.history_same_rows[i]}\n"

            # 添加错误数据和路径分析
            withhistory_prompt += f"Incorrect Generated Tabular Data: \n"
            if i < len(self.all_questions_path_samples) and self.all_questions_path_samples[i]:
                for path_str, path_data in self.all_questions_path_samples[i].items():
                    samples = path_data['samples']
                    for sample in samples:
                        withhistory_prompt += f"{list(sample['original_data'].values())} "
                    withhistory_prompt += "\n"
                    withhistory_prompt += f"The labels(the last column) of these samples are wrong. Their prediction path in model is: {path_str}\n"
            else:
                withhistory_prompt += f"No incorrect samples for question {i + 1}\n"

        return withhistory_prompt

    def _evaluate_round(self):
        """评估当前轮次的效果"""
        print("===========Evaluating Round===========")
        evaluation_results = {}

        for model_id, df in self.shared_model_data.items():
            if len(df) > 0:
                result = evaluate_singleModel(path=self.dataset_name, additional_data=df.values, model_id=model_id)
                evaluation_results[model_id] = result
                print(f"Model {model_id}: {len(df)} samples")

        return evaluation_results

    def run_multi_round_generation(self, num_rounds=3, initial_questions=3, target_id=2):
        """运行多轮反思生成"""
        print(f"Starting {num_rounds} rounds of reflective data generation")

        # 第一轮：初始数据生成
        initial_result = self.initial_data_generation(initial_questions)

        # 后续轮次：反思数据生成
        for round_num in range(1, num_rounds):
            print(f"\n{'=' * 60}")
            print(f"Starting Round {round_num + 1} of {num_rounds}")
            print(f"Current history examples: {len(self.history_question)}")
            print(f"{'=' * 60}")

            reflective_result = self.reflective_data_generation(model_id=target_id)
            if reflective_result is None:
                print(f"Round {round_num + 1} failed, stopping.")
                break

            print(f"After round {round_num + 1}, total history examples: {len(self.history_question)}")

            # 可以在这里添加早停条件
            # if self._should_stop():
            #     break

        return self.get_summary()

    def get_summary(self):
        """获取总结信息"""
        summary = {
            'total_rounds': self.current_round,
            'total_data_generated': sum(len(df) for df in self.all_dfs),
            'total_history_examples': len(self.history_question),
            'shared_model_data': {k: len(v) for k, v in self.shared_model_data.items()},
            'round_history': self.round_history,
            'final_evaluation': self._evaluate_round()
        }

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total rounds completed: {summary['total_rounds']}")
        print(f"Total data generated: {summary['total_data_generated']}")
        print(f"Total history examples: {summary['total_history_examples']}")
        print(f"Shared model data: {summary['shared_model_data']}")

        return summary

    def checkScale(self, model_id):
        if self.shared_model_data[model_id].shape[1] != 8:
            return False
        else:
            return True


# 使用示例
def main():
    # 配置参数
    dataset_name = "bank-marketing"
    data_path = "dataset/clf_num/bank-marketing.csv"
    json_path = "current_block/bank-marketing_ROU=0.2_f1-error_block.json"
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

    # 运行多轮生成
    summary = generator.run_multi_round_generation(num_rounds=3, initial_questions=3, target_id=2)

    return summary


if __name__ == "__main__":
    summary = main()
