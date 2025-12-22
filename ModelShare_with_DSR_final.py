import csv
import itertools
import os
import re
import sys
import time
import pickle
from pathlib import Path

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# from queue import PriorityQueue
import queue
import heapq
import json

from tqdm import tqdm


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


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.count = itertools.count()  # 创建一个计数器

    def is_empty(self):
        return len(self.heap) == 0

    def put(self, rule):
        # 使用 -rule.index 使得 index 越大优先级越高
        # 使用 self.count 实现先进先出（LIFO）
        count = next(self.count)
        heapq.heappush(self.heap, (-rule.index, count, rule))

    def get(self):
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")
        return heapq.heappop(self.heap)[2]  # 返回优先级最高的元素

    def __iter__(self):
        # 使用浅拷贝的堆进行迭代，以避免破坏原始队列
        copy_heap = self.heap[:]
        heapq.heapify(copy_heap)  # 确保拷贝的堆有效
        while copy_heap:
            yield heapq.heappop(copy_heap)[2]  # 返回优先级最高的元素


class Model:
    def __init__(self, data):
        self.data = np.array(data)
        self.model = DecisionTreeClassifier(criterion='entropy', splitter='best')
        self.rf_mdoel = RandomForestClassifier(criterion="entropy")
        self.id = 0

    def train(self):
        X = self.data[:, :-1]
        y = self.data[:, -1]
        self.model.fit(X, y)
        # m.data[:. :-1] + different_rows->新训练集
        # m.data[:,-1]->测试集
        # m.model.fit(X_new, Y_new)
        # 解析m.model路径
        # 对比原来Ma的路径
        # 拿到第一个不一致的节点
        # 节点融入prompt（feedback）

    def train_RF(self):
        X = self.data[:, :-1]
        y = self.data[:, -1]
        self.rf_mdoel.fit(X, y)

    # 用随机森林去算指标
    def predict(self, instance):
        return (self.rf_mdoel).predict_proba([instance[:-1]])[0][-1]
        # return (self.model).predict_proba([instance[:-1]])[0][-1]

    def get_model_id(self):
        return self.id

    def get_model(self):
        return self.model

    def get_rf_model(self):
        return self.rf_mdoel

    def get_probability(self, input):
        try:
            return self.rf.classifyInstance(input)  # 获取实例的分类概率
        except Exception as e:
            print(e)
            return -1

    def set_model_id(self, num):
        self.id = num

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        if isinstance(other, Model):
            return id(self) == id(other)

        # def __eq__(self, other):
        #     if isinstance(other, Model):
        #         return other.rf == self.rf   # 比较两个模型是否相等
        return False


class Predicate:
    def __init__(self, first, second, third):
        self.first = first
        self.second = second
        self.third = third

    def get_first(self):
        return self.first

    def get_second(self):
        return self.second

    def get_third(self):
        return self.third

    def __str__(self):
        return f"({self.first} {self.second} {self.third})"

    def __eq__(self, other):
        if isinstance(other, Predicate):
            return self.first == other.get_first() and self.second == other.get_second() and self.third == other.get_third()
        return False

    def __hash__(self):
        return hash(self.first) + hash(self.second) + hash(self.third)


class Rule:
    def __init__(self, index, context):
        self.index = index
        self.context = context

    def set_context(self, context):
        self.context = context

    def get_context(self):
        return self.context

    def set_index(self, index):
        self.index = index

    def get_index(self):
        return self.index

    def __lt__(self, other):
        # 当前规则的index更大，优先级更高
        return self.index > other.index

    def __str__(self):
        context_str = ', '.join(str(pred) for pred in self.context)
        return f"[index={self.index}, context={{ {context_str} }}]"

    def add_constraint(self, constraint):
        self.context.add(constraint)


class ModelShare:
    # 存储modelID；最初建立模型时的训练数据（用于测试新生成的数据是否符合）；内部包含的规则list；最终用于训练testData时合并相同id的block后形成的final_test与final_train
    def save_models(self, data, model_id):
        """Save all models in model_set to disk using pickle.

        Each model is saved as 'model/{dataset_name}_model_{model_id}.pkl'
        """
        filename = self.model_dir / f"{self.dataset}_{rou}_model_{model_id}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved model to {filename}")

    def load_model(self, model_id):
        """Load a model from disk using pickle.

        Args:
            model_id: ID of the model to load

        Returns:
            The loaded model or None if not found
        """
        filename = self.model_dir / f"{self.dataset}_model_{model_id}.pkl"
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Model {model_id} not found")
            return None

    def __init__(self, dataset_name, rou):
        self.model_set = set()
        self.class_index = -1  # assuming the class index is the last column
        self.RouMax = rou  # here
        self.blocksize = 100
        self.num_in_block = 0
        self.block_num = 0
        self.modelID = 0
        self.total_error = 0
        self.dataset = dataset_name
        # Create model directory if it doesn't exist
        self.model_dir = Path("model")
        self.model_dir.mkdir(exist_ok=True)
        self.output_path = "block/" + dataset_name + "_ROU=" + str(rou) + "_f1-error" + "_block.csv"
        self.current_block = "current_block/" + dataset_name + "_ROU=" + str(rou) + "_f1-error" + "_block.json"

    def convert_ndarray_to_list(self, obj):
        """递归转换字典/列表中的 numpy.ndarray 为 list"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_ndarray_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray_to_list(item) for item in obj]
        else:
            return obj

    def get_final_block_and_rule(self, blockpath, rulepath, merge_block_dict, merge_rule_dict):
        '''
        -outpath: 最终划分的block输出路径
        -merge_block_dict: 存放modelid和block的字典
        -merge_rule_dict: 存放modelid和rule的字典
        '''
        merge_block_dict = self.convert_ndarray_to_list(merge_block_dict)
        with open(blockpath, 'w', encoding='utf-8') as block_f:
            json.dump(merge_block_dict, block_f, indent=4, ensure_ascii=False)

        new_rulesets = []
        for key, values in merge_rule_dict.items():
            new_values = []
            # values中按照规则间析取，规则内合取，values的形式是一个可能多个rule的列表，实际上这个列表表示同属于一个sharemodel的规则集
            for rule in values:
                predset = rule.get_context()
                str_predset = [str(pred) for pred in predset]
                new_values.append(str_predset)

            print(new_values)
        with open(rulepath, 'w', encoding='utf-8') as block_f:
            json.dump(merge_rule_dict, block_f, indent=4, ensure_ascii=False)

    def get_current_block_and_rule(self, train_data_df: pd.DataFrame, block_data, rule, save_path, modelID: int,
                                   validScore: float):

        # 读取原始 JSON 数据
        try:
            with open(save_path, 'r', encoding='utf-8') as block_f:
                data = json.load(block_f)
        except FileNotFoundError:
            data = {}  # 如果文件不存在，则初始化为空字典

        rule_key = ','.join([str(pred) for pred in rule.get_context()])

        data_df = pd.DataFrame(block_data, columns=train_data_df.columns)
        # 为data添加临时索引列
        tmp_df = train_data_df.copy()
        tmp_df['temp_index'] = train_data_df.index
        # 合并数据并提取原索引
        merged = tmp_df.merge(data_df)
        indices = merged['temp_index'].tolist()

        # 追加新的数据和规则支持度
        block_data_list = [value.tolist() if isinstance(value, np.ndarray) else value for value in block_data]
        # 存储规则支持度和对应的数据
        data[rule_key] = {
            "modelID": modelID,
            "support": rule.get_index(),  # 添加规则支持度
            "validScore": validScore,
            "data": block_data_list  # 原始数据
        }

        # 将修改后的数据重新写入文件
        with open(save_path, 'w', encoding='utf-8') as block_f:
            json.dump(data, block_f, indent=4, ensure_ascii=False)

    def add_pred(self, rule, pred):
        temp = rule.get_context()
        res = set(temp)
        res.add(pred)
        return res

    def calc_deltr(self, data, model):
        max_val = 0
        min_val = 1
        for ins in data:
            minus = abs(ins[-1] - model.predict(ins))
            if minus < min_val:
                min_val = minus
            if minus > max_val:
                max_val = minus
        return (min_val + max_val) / 2

    # 使用F1-score作为dlter的值
    def calc_max_error(self, data, model):
        data_x = [row[:-1] for row in data]
        data_y = [row[-1] for row in data]
        y_pred = (model.get_rf_model()).predict(data_x)
        # y_pred = (model.get_model()).predict(data_x)
        try:
            acc = accuracy_score(data_y, y_pred)
            error = 1 - acc
            return error
        except UndefinedMetricWarning:
            # 0825修改，f1 score未定义，说明模型预测值全为0
            return 0

    # 用决策树划分规则
    def produce_pred(self, rf):
        predicate_set = []
        nodes = queue.Queue()
        nodes.put(0)
        tree = rf.tree_
        while not nodes.empty():
            node_id = nodes.get()
            if tree.children_left[node_id] != -1:
                feature = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                predicate_set.append(Predicate(self.attr_name[feature], ">=", threshold))
                predicate_set.append(Predicate(self.attr_name[feature], "<", threshold))
                nodes.put(tree.children_left[node_id])
                nodes.put(tree.children_right[node_id])
                if len(predicate_set) > 50:
                    return predicate_set
        return predicate_set

    def select_by_rule(self, all_data, rule):
        # Apply the rule to filter data
        # Placeholder implementation
        predset = rule.get_context()
        data = []
        if len(predset) != 0:
            for element in all_data:
                # flag一直为0说明一直遵守rule，违反的不被存储。
                flag = 0
                for pred in predset:
                    # attr下标
                    loc = 0
                    for attr in self.attr_name:
                        first = pred.get_first()
                        if attr == first:
                            # 被约束的属性匹配
                            if pred.get_second() == "<=":
                                if element[loc] > pred.get_third():
                                    # 不满足约束
                                    flag = 1
                                    break
                            elif pred.get_second() == ">=":
                                if element[loc] < pred.get_third():
                                    # 不满足约束
                                    flag = 1
                                    break
                            elif pred.get_second() == "<":
                                if element[loc] > pred.get_third():
                                    flag = 1
                                    break
                            elif pred.get_second() == ">":
                                if element[loc] < pred.get_third():
                                    flag = 1
                                    break
                            elif pred.get_second() == "=":
                                if element[loc] != pred.get_third():
                                    flag = 1
                                    break
                        loc += 1
                    if flag == 1:
                        break
                if flag == 0:
                    data.append(element)
        else:
            data = all_data
        return data  # Replace with actual filtering logic

    def select_by_rule_for_test(self, all_data, rule):
        # Apply the rule to filter data
        # Placeholder implementation
        self.attr_name = all_data.columns.values.tolist()
        all_data = all_data.values.tolist()
        predset = rule.get_context()
        data = []
        if len(predset) != 0:
            for element in all_data:
                # flag一直为0说明一直遵守rule，违反的不被存储。
                flag = 0
                for pred in predset:
                    # attr下标
                    loc = 0
                    for attr in self.attr_name:
                        first = pred.get_first()
                        if attr == first:
                            # 被约束的属性匹配
                            if pred.get_second() == "<=":
                                if element[loc] > pred.get_third():
                                    # 不满足约束
                                    flag = 1
                                    break
                            elif pred.get_second() == ">=":
                                if element[loc] < pred.get_third():
                                    # 不满足约束
                                    flag = 1
                                    break
                            elif pred.get_second() == "<":
                                if element[loc] > pred.get_third():
                                    flag = 1
                                    break
                            elif pred.get_second() == ">":
                                if element[loc] < pred.get_third():
                                    flag = 1
                                    break
                            elif pred.get_second() == "=":
                                if element[loc] != pred.get_third():
                                    flag = 1
                                    break
                        loc += 1
                    if flag == 1:
                        break
                if flag == 0:
                    data.append(element)
        else:
            data = all_data
        return data  # Replace with actual filtering logic

    def test_pri(self):
        i = 1
        pri = self.priority_queue
        while i < 5:
            temp = Predicate("a", "b", i)
            temp_1 = set()
            temp_1.add(temp)
            pri.put(Rule(0, temp_1))
            i += 1
        while pri.is_empty() != True:
            rule = pri.get()
            x = rule.get_context()
            for x1 in x:
                print(x1.get_third())

    def read_data(self, path):
        # with open(path, newline='') as csvfile:
        #     datareader = csv.reader(csvfile)
        #     self.attr_name = next(datareader)
        #     data = [list(row) for row in datareader]
        data = pd.read_csv(path)
        columns = data.columns
        self.attr_name = data.columns.values.tolist()
        data[self.attr_name[-1]] = data[self.attr_name[-1]].astype('int64')
        # print(data[self.attr_name[-1]].dtype)  # 查看该列的数据类型
        # print(data.iloc[0,-1])
        data = data.values.tolist()
        # print(data[0][-1])
        return data, columns

    def crr(self, data_path):
        offline_saved = []  # List to store all saveData dictionaries in order
        total_flag = 0
        start = time.time()
        priorityqueue = PriorityQueue()
        priorityqueue.put(Rule(0, set()))

        score_output_path = "score_output/test/" + self.dataset + "_ROU=" + str(
            self.RouMax) + "_f1-error" + "_score_output.txt"

        all_data, columns = self.read_data(data_path)

        total_size = len(all_data)

        # 将总数据按照4：1划分为训练数据和测试数据
        train_size = int(total_size * 0.8)

        total_data_array = np.array(all_data)

        X_all = total_data_array[:, :-1]
        y_all = total_data_array[:, -1]
        # 这里划分的是训练集和测试集合（最终用于划分的测试集）
        total_X_train, total_X_val, total_y_train, total_y_val = train_test_split(X_all, y_all, test_size=0.2,
                                                                                  random_state=42)
        # 首先，将 y_train_all 转换为二维数组（列向量）

        y_train_2d_all = total_y_train.reshape(-1, 1)  # -1 表示让NumPy自动计算这一维的大小
        y_val_2d_all = total_y_val.reshape(-1, 1)  # -1 表示让NumPy自动计算这一维的大小

        # 然后，使用 hstack 水平堆叠 X_train 和 y_train_2d
        train_data = np.hstack((total_X_train, y_train_2d_all))
        test_data = np.hstack((total_X_val, y_val_2d_all))

        test_data_df = pd.DataFrame(test_data, columns=columns).to_csv('filtered_results/test.csv', index=False)
        train_data_df = pd.DataFrame(train_data, columns=columns)

        # merge_block的key表示modelID，value表示对应modelID的block数据。 merge_rule的Key表示modelID，value表示对应modelID的规则
        merge_block_dict = {}
        merge_rule_dict = {}

        # 打开一个文件用于写入
        f = open(self.output_path, 'w')

        # 保存原来的stdout
        old_stdout = sys.stdout

        # 将stdout重定向到文件
        sys.stdout = f

        # 先将整个训练数据训练一遍，针对test集剩一块的情况
        model_final_part = DecisionTreeClassifier(criterion='entropy', splitter="best")
        model_final_part.fit(train_data[:, :-1], train_data[:, -1])
        while not priorityqueue.is_empty():

            tmp_block_data = []
            tmp_rule_list = []
            end = time.time()
            # if end - start > 3600 or len(train_data) <= train_size * 0.2:
            #     break
            if end - start > 7200 or len(train_data) <= train_size * 0.01:
                break
            rule = priorityqueue.get()
            data = self.select_by_rule(train_data, rule)
            if len(data) <= 1:
                continue
            flag = 0
            share_rou = 0.0
            shared_model = Model(data)
            for model in self.model_set:
                # 如果error不能被容忍，flag = 1
                flag = 0
                deltr = self.calc_deltr(data, model)
                # for ins in data:
                #     rou = abs(ins[-1] - (model.predict(ins) + deltr))
                #     if rou >= self.RouMax:
                #         flag = 1
                #     if rou > share_rou:
                #         share_rou = rou
                # 使用1 - f1-score来作为评判标准
                temp_rou = self.calc_max_error(data, model)
                if temp_rou >= self.RouMax:
                    flag = 1
                if temp_rou > share_rou:
                    share_rou = temp_rou
                if flag == 0:
                    shared_model = model
                    break
            if flag == 0 and self.model_set:
                key = shared_model.get_model_id()
                # 获取当前modelID对应的block数据，并将新的block连接在之前的block后面
                tmp_block_data = merge_block_dict.get(key)
                tmp_block_data = tmp_block_data + data
                merge_block_dict[key] = tmp_block_data

                tmp_rule_list = merge_rule_dict.get(shared_model.get_model_id())
                tmp_rule_list.append(rule)
                merge_rule_dict[key] = tmp_rule_list

                # 这里有点问题，如果按照源代码deltr必须为0，1才有意思，但是计算deltr的过程deltr很有可能是分数
                # rule.add_constraint(Predicate("label", "=", deltr))
                self.total_error += share_rou

                # if self.num_in_block == 0:
                # tmp_data = np.array(data)
                # X = tmp_data[:, :-1]  # 获取除了最后一列之外的所有列作为特征
                # y = tmp_data[:, -1]   # 获取最后一列作为标签
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                # tmp_model = shared_model.get_model()
                # tmp_model.fit(X_train,y_train)
                # # 使用模型进行预测
                # y_pred = tmp_model.predict(X_test)
                # f1 = f1_score(y_test, y_pred)
                # print(f"modelID : {shared_model.get_model_id()} F1 Score: {f1}")
                print(f"modelID : {shared_model.get_model_id()}")
                data_size = len(data)
                self.num_in_block += data_size
                # if self.num_in_block > self.blocksize:
                #     self.block_num += 1
                #     self.num_in_block = 0
                self.block_num += 1

                # 获取规则和block
                self.get_current_block_and_rule(train_data_df, data, rule, save_path=self.current_block,
                                                modelID=shared_model.get_model_id(), validScore=1 - share_rou)

                # 输出数据
                for i in range(len(data)):
                    # 遍历每一行的每一个元素
                    for j in range(len(data[i]) - 1):  # 减1是因为不想在最后一个元素后添加逗号
                        print(data[i][j], end=',')  # 使用end参数来避免自动换行，并在元素后添加逗号
                    # 打印最后一个元素（标签）后换行
                    print(data[i][-1])  # 使用-1索引来获取最后一个元素
                data = np.array(data)
                train_data = np.atleast_2d(train_data)
                data = np.atleast_2d(data)
                # print("train_data shape:", train_data.shape)
                # print("data shape:", data.shape)
                # 创建一个布尔掩码，标识 train_data 中的行是否在 data 中存在
                mask = np.array(
                    [np.all(np.any(np.all(train_data[i] == data, axis=1), axis=0)) for i in range(train_data.shape[0])])

                # 反转掩码，保留那些不在 data 中的行
                train_data = train_data[~mask]
            else:
                index = 0
                max_index = 0
                for model in self.model_set:
                    sum_ = sum(1 for ins in data if abs(ins[-1] - (model.predict(ins) + deltr)) <= self.RouMax)
                    index = sum_ / len(data)
                    if index > max_index:
                        max_index = index
                        rule.set_index(max_index)
                # 取出一部分训练数据作为校准集去计算deltr，避免训练数据再次被预测
                data_array = np.array(data)

                X = data_array[:, :-1]
                y = data_array[:, -1]
                # 划分验证集
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
                # 首先，将 y_train 转换为二维数组（列向量）
                y_train_2d = y_train.reshape(-1, 1)  # -1 表示让NumPy自动计算这一维的大小
                y_val_2d = y_val.reshape(-1, 1)  # -1 表示让NumPy自动计算这一维的大小

                # 然后，使用 hstack 水平堆叠 X_train 和 y_train_2d
                tmp_train_data = np.hstack((X_train, y_train_2d))
                val_data = np.hstack((X_val, y_val_2d))
                new_model = Model(tmp_train_data)

                # 训练决策树的目的是去划分规则，保证每次运行的结果统一
                new_model.train()

                # 使用随机森林去预测，提高效果
                new_model.train_RF()
                # bad_instances = []
                max_error = 0
                deltr = self.calc_deltr(val_data, new_model)
                # 可以优化，可以将self.calc_deltr(data, new_model)中的max——value返回，避免下面的循环，其实max——error就是max_value
                # for ins in data:
                #     cha = ins[-1] - new_model.predict(ins)
                #     # if abs(cha - deltr) > self.RouMax:
                #     #     bad_instances.append(ins) #没有用到bad_instances
                #     error = abs(cha)
                #     if error > max_error:
                #         max_error = error
                max_error = self.calc_max_error(val_data, new_model)
                # attention : info maxError <= RouMax
                if (self.total_error + max_error) <= self.RouMax * (self.block_num + 1):
                    new_model.set_model_id(self.modelID)

                    merge_block_dict[self.modelID] = data
                    tmp_rule_list.append(rule)
                    merge_rule_dict[self.modelID] = tmp_rule_list

                    self.model_set.add(new_model)
                    saveData = {
                        'modelID': self.modelID,  # int
                        'original_trainData': tmp_train_data,  # 可以用于训练数据
                        'ruleList': None,  # list
                        'final_testData': None,
                        'final_trainData': None,
                        'validScore': max_error
                    }
                    offline_saved.append(saveData)  # Add to offline_saved list
                    # self.save_models(saveData)
                    self.total_error += max_error
                    self.modelID += 1

                    # save block
                    # if self.num_in_block == 0:
                    #     tmp_data = np.array(data)
                    #     X = tmp_data[:, :-1]  # 获取除了最后一列之外的所有列作为特征
                    #     y = tmp_data[:, -1]   # 获取最后一列作为标签
                    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    #     tmp_model = new_model.get_model()
                    #     tmp_model.fit(X_train,y_train)

                    #     # 使用模型进行预测
                    #     y_pred = tmp_model.predict(X_test)
                    #     f1 = f1_score(y_test, y_pred)
                    #     print(f"modelID : {shared_model.get_model_id()},F1 Score: {f1}")
                    print(f"modelID : {new_model.get_model_id()}")
                    self.num_in_block += len(data)
                    # if self.num_in_block > self.blocksize:
                    #     self.block_num += 1
                    #     self.num_in_block = 0
                    self.block_num += 1

                    self.get_current_block_and_rule(train_data_df, data, rule, save_path=self.current_block,
                                                    modelID=new_model.get_model_id(), validScore=1 - max_error)

                    # 输出数据
                    for i in range(len(data)):
                        # 遍历每一行的每一个元素
                        for j in range(len(data[i]) - 1):  # 减1是因为不想在最后一个元素后添加逗号
                            print(data[i][j], end=',')  # 使用end参数来避免自动换行，并在元素后添加逗号
                        # 打印最后一个元素（标签）后换行
                        print(data[i][-1])  # 使用-1索引来获取最后一个元素
                    data = np.array(data)
                    if np.array_equal(train_data, data):
                        break
                        # 确保 train_data 和 data 至少是二维数组
                    train_data = np.atleast_2d(train_data)
                    data = np.atleast_2d(data)
                    # print("train_data shape:", train_data.shape)
                    # print("data shape:", data.shape)
                    # 创建一个布尔掩码，标识 train_data 中的行是否在 data 中存在
                    mask = np.array([np.all(np.any(np.all(train_data[i] == data, axis=1), axis=0)) for i in
                                     range(train_data.shape[0])])

                    # 反转掩码，保留那些不在 data 中的行
                    train_data = train_data[~mask]
                else:
                    pred_set = self.produce_pred(new_model.get_model())
                    for pred in pred_set:
                        r = Rule(max_index, self.add_pred(rule, pred))
                        count = 0
                        for itor in priorityqueue:
                            if itor.get_context() == r.get_context():
                                count = 1
                                itor.set_index(r.get_index())
                                break
                        if count == 0:
                            priorityqueue.put(r)
                max_error += 1
        f.close()
        end_final = time.time()
        # sys.stdout = old_stdout
        f = open(score_output_path, "w")
        sys.stdout = f
        print(f"time: {end_final - start}")
        print(f"模型的数量：{len(merge_block_dict)}")

        y_test_all = []
        y_test_all = np.array(y_test_all)
        y_pre_all = []
        y_pre_all = np.array(y_pre_all)

        final_block_fold = "final_block/"
        if not os.path.exists(final_block_fold):
            os.mkdir(final_block_fold)
        final_block_path = final_block_fold + self.dataset + "_ROU=" + str(self.RouMax) + "_f1-error" + "_block.json"
        final_rule_fold = "final_rule/"
        if not os.path.exists(final_rule_fold):
            os.mkdir(final_rule_fold)
        final_rule_path = final_rule_fold + self.dataset + "_ROU=" + str(self.RouMax) + "_f1-error" + "_rule.json"

        # self.get_final_block_and_rule(blockpath=final_block_path, rulepath=final_rule_path, merge_block_dict=merge_block_dict, merge_rule_dict=merge_rule_dict)
        # 开始测试模型 ，规则列表从 merge_rule_dict中取，规则内合取，规则间析取,用id计数当前modelID，方便存储到offline
        test_data_copy = test_data.copy()

        # ==== 在主程序最前面，先定义一个列表来存储各轮结果 ====
        per_round_selected = []
        for id, test_key in enumerate(merge_block_dict.keys()):
            final_traindata = merge_block_dict.get(test_key)
            # 本轮用的数据，最后筛选完了再更新
            # 当前id下的所有rule
            tmp_rule = merge_rule_dict.get(test_key)

            # 1) 从 test_data 中剔除之前所有轮已选的行
            if per_round_selected:
                # 如果你用列表存，每次把所有已选DataFrame concat后 drop
                all_prev = pd.concat(per_round_selected, ignore_index=False)
                # test_data_copy = test_data_copy.drop(all_prev.index, errors='ignore')
                test_data_copy_df = pd.DataFrame(test_data_copy, columns=columns)
                test_data_copy_df = test_data_copy_df.drop(this_round_indices, errors='ignore')
                test_data_copy = test_data_copy_df.values  # 转换回 numpy 数组
            # 如果用字典存，也可以合并 dict.values()

            # 2) 找到本轮所有 target_id 对应的规则
            target_rules = tmp_rule

            # 3) 按规则筛选
            this_round_indices = []
            for rule_str in target_rules:
                preds = parse_rule_string(str(rule_str))
                rule = Rule(0, set(preds))
                # select_by_rule_for_test 返回 list of lists
                test_data_copy_df = pd.DataFrame(test_data_copy, columns=columns)
                selected_list = self.select_by_rule_for_test(test_data_copy_df, rule)
                # selected_list = self.select_by_rule_for_test(test_data_copy, rule)
                if not selected_list:
                    continue
                # 转回 DataFrame
                # selected_df = pd.DataFrame(selected_list, columns=test_data.columns)
                test_data_df_temp = pd.DataFrame(test_data, columns=columns)
                selected_df = pd.DataFrame(selected_list, columns=test_data_df_temp.columns)
                # 定位索引：用 tuple 比对最快
                tup_set = set(selected_df.apply(tuple, axis=1))
                # mask = test_data_copy.apply(tuple, axis=1).isin(tup_set)
                test_data_copy_df = pd.DataFrame(test_data_copy, columns=columns)
                mask = test_data_copy_df.apply(tuple, axis=1).isin(tup_set)
                # idxs = test_data_copy[mask].index.tolist()
                idxs = test_data_copy_df[mask].index.tolist()  # 使用 DataFrame 版本
                this_round_indices.extend(idxs)

            # 4) 把本轮所有 idx 去重，然后取出对应行
            this_round_indices = list(dict.fromkeys(this_round_indices))  # 保序去重
            # this_df = test_data_copy.loc[this_round_indices]
            this_df = test_data_copy_df.loc[this_round_indices]

            # 5) 存入 per_round_selected
            per_round_selected.append(this_df)
            # 如果用 dict： per_round_selected[target_id] = this_df

            # 6) 从 test_data 中删除本轮已选行
            # test_data_copy = test_data_copy.drop(this_round_indices, errors='ignore')
            test_data_copy_df = pd.DataFrame(test_data_copy, columns=columns)
            test_data_copy_df = test_data_copy_df.drop(this_round_indices, errors='ignore')
            test_data_copy = test_data_copy_df.values  # 转换回 numpy 数组

            # 7) 编辑final_testdata
            # final_testdata = list(this_df)
            final_testdata = this_df.values.tolist()

            # Update offline_saved[0] with testData and rule
            if offline_saved:  # Check if offline_saved is not empty
                offline_saved[id]['final_testData'] = final_testdata  # Update testData
                offline_saved[id]['final_trainData'] = final_traindata
                offline_saved[id]['ruleList'] = tmp_rule  # Update rule with the first rule from sorted list
                self.save_models(offline_saved[id], id)
            # 开始用同一个sharemodel的Block数据取训练数据
            X_train = [row[:-1] for row in final_traindata]
            y_train = [row[-1] for row in final_traindata]
            X_test = [row[:-1] for row in final_testdata]
            print(f"X_test 长度: {len(X_test)}")
            if len(X_test) > 0:
                print(f"X_test[0] 类型: {type(X_test[0])}")
                print(f"X_test[0] 内容: {X_test[0]}")
                # 检查是否有字符串数据
                for i, item in enumerate(X_test[0]):
                    if isinstance(item, str):
                        print(f"发现字符串在第 {i} 个位置: {item}")

            y_test = [row[-1] for row in final_testdata]

            final_model = RandomForestClassifier(criterion='entropy')
            # final_model = DecisionTreeClassifier(criterion='entropy', splitter="best")

            final_model.fit(X_train, y_train)

            y_test = np.array(y_test)

            X_test = np.array(X_test)
            if X_test.size == 0:
                continue
            if X_test.size == 1:
                X_test = X_test.reshape(1, -1)
            y_pred = final_model.predict(X_test)
            y_test_all = np.hstack((y_test_all, y_test))
            y_pre_all = np.hstack((y_pre_all, y_pred))
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            print(f"modelID: {test_key}")
            for the_tmp_rule in tmp_rule:
                r: Rule = the_tmp_rule
                print(f"rule: {(r.get_context())}")
            print(f"trainnum: {len(final_traindata)}")
            print(f"testnum: {len(final_testdata)}")
            print(f"f1: {f1}")
            print(f"accuracy: {accuracy}")
            print(f"precision: {precision}")
            print("----------------------------------------------------------------------------")
            print("\n")

        # 如果testdata未被划分完全，则是用整个数据训练的模型去预测
        if len(test_data) != 0:
            if offline_saved:  # Check if offline_saved is not empty
                offline_saved[-1]['final_testData'] = test_data  # Update testData
                # 所有traindata,在model_final_part中
                offline_saved[-1]['final_trainData'] = train_data
                # 所有rule
                offline_saved[-1]['ruleList'] = []  # Update rule with the first rule from sorted list
                self.save_models(offline_saved[-1], -1)

            test_data = np.array(test_data)
            X_test = test_data[:, :-1]
            y_test = test_data[:, -1]
            if X_test.size == 1:
                X_test.reshape(1, -1)
            y_pred_final = model_final_part.predict(X_test)
            y_test_all = np.hstack((y_test_all, y_test))
            y_pre_all = np.hstack((y_pre_all, y_pred_final))
        final_f1 = f1_score(y_test_all, y_pre_all, zero_division=0)
        final_pre = precision_score(y_test_all, y_pre_all)
        final_recall = recall_score(y_test_all, y_pre_all)
        final_accuracy = accuracy_score(y_test_all, y_pre_all)
        print("\n")
        print("---------------------------------------------------------------------------------")
        print(f"total_f1: {final_f1}")
        print(f"total_precision: {final_pre}")
        print(f"total_recall: {final_recall}")
        print(f"total_accuracy: {final_accuracy}")
        print(f"testnum: {len(y_test_all)}")
        print(f"prenum: {len(y_pre_all)}")
        f.close()
        sys.stdout = old_stdout
        print("子集划分完毕")
        # print(f"rule: {tmp_rule}")
        return final_f1, final_accuracy

    # 测试优先级队列当优先级相同时是否按照先进先出的规则
    # if __name__ == "__main__":
    #     pq = PriorityQueue()
    #     rules = [Rule(1, "rule1"), Rule(1, "rule2"), Rule(2, "rule3"), Rule(1, "rule4")]

    #     for rule in rules:
    #         pq.put(rule)

    #     while not pq.is_empty():
    #         rule = pq.get()
    #         print(rule)

    def test_generated_block(self, train_path, test_path, rule):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        name = train_path.split('/')[-1]
        output = 'filtered_results/' + name.split('.')[0] + '_预测对比.csv'

        test_out = 'filtered_results/' + name.split('.')[0] + '_测试集.csv'
        # test_data = test_data.values.tolist()
        final_test_data = self.select_by_rule_for_test(test_data, rule=rule)
        pd.DataFrame(final_test_data, columns=train_data.columns).to_csv(test_out, index=False)
        final_test_data = np.array(final_test_data)
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]

        X_test = final_test_data[:, :-1]
        y_test = final_test_data[:, -1]

        model = RandomForestClassifier(random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_duibi = [y_test, y_pred]
        y_duibi = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
        y_duibi.to_csv(output, index=False)

        final_f1 = f1_score(y_test, y_pred, zero_division=0)
        final_pre = precision_score(y_test, y_pred)
        final_recall = recall_score(y_test, y_pred)
        final_accuracy = accuracy_score(y_test, y_pred)
        print("\n")
        print("---------------------------------------------------------------------------------")
        print(f"total_f1: {final_f1}")
        print(f"total_precision: {final_pre}")
        print(f"total_recall: {final_recall}")
        print(f"total_accuracy: {final_accuracy}")
        print(f"testnum: {len(y_test)}")
        print(f"trainnum: {len(y_train)}")


if __name__ == "__main__":
    datasetlist = ["eye_movements", ]
    for i in tqdm(range(0, len(datasetlist))):
        dataset_name = datasetlist[i]
        rou = 0.01
        model_share = ModelShare(dataset_name, rou)
        model_share.RouMax = rou
        path = "dataset/clf_num/" + dataset_name + ".csv"
        f1 = model_share.crr(path)

        rou = 0.025
        model_share_2 = ModelShare(dataset_name, rou)
        model_share_2.RouMax = rou
        f1_2 = model_share_2.crr(path)

        rou = 0.05
        model_share_3 = ModelShare(dataset_name, rou)
        model_share_3.RouMax = rou
        f1_3 = model_share_3.crr(path)

        rou = 0.075
        model_share_4 = ModelShare(dataset_name, rou)
        model_share_4.RouMax = rou
        f1_4 = model_share_4.crr(path)

        rou = 0.1
        model_share_5 = ModelShare(dataset_name, rou)
        model_share_5.RouMax = rou
        f1_5 = model_share_5.crr(path)
