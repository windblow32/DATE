import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from ModelShare_with_DSR_final import Model, Rule, Predicate


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


def evaluate_singleModel(path, additional_data, model_id):
    if additional_data is None:
        print(f"No shared data for model {model_id}.")
        return 0
    print(f"model id: {model_id}")
    saveData = load_model(path, model_id)
    final_trainData = pd.DataFrame(saveData['final_trainData'])
    final_testData = pd.DataFrame(saveData['final_testData'])
    additional_df = pd.DataFrame(additional_data, columns=final_trainData.columns)

    # Concatenate them row-wise
    combined_data = pd.concat([final_trainData, additional_df], ignore_index=True)

    X_train_expand = combined_data.iloc[:, :-1]  # 所有列除了最后一列
    y_train_expand = combined_data.iloc[:, -1]  # 最后一列作为标签

    X_test = final_testData.iloc[:, :-1]
    y_test = final_testData.iloc[:, -1]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_expand, y_train_expand)

    y_pred = model.predict(X_test)

    # 计算F1分数
    # expand_f1 = f1_score(y_test, y_pred)
    # 计算错误率
    expand_error = 1 - accuracy_score(y_test, y_pred)

    X_train_origin = final_trainData.iloc[:, :-1]
    y_train_origin = final_trainData.iloc[:, -1]
    model_origin = RandomForestClassifier(random_state=42)
    model_origin.fit(X_train_origin, y_train_origin)

    y_pred_origin = model_origin.predict(X_test)

    # 计算F1分数
    # origin_f1 = f1_score(y_test, y_pred_origin)
    # 计算错误率
    origin_error = 1 - accuracy_score(y_test, y_pred_origin)

    print(f'Error Rate: {expand_error}')
    print(f'origin Error Rate: {origin_error}')
    if origin_error != 0:
        err_percent = 1 - expand_error / origin_error
    else:
        err_percent = 0
    print(f"error rate improve: {100 * err_percent}")
    # print(f'F1 Score: {expand_f1}')
    # print(f'origin F1 Score: {origin_f1}')
    # f1_gain = expand_f1 - origin_f1
    # print(f"f1 rate improve: {f1_gain}")
    return origin_error - expand_error


def evaluate_validation(path, additional_data, model_id):
    saveData = load_model(path, model_id)
    # 先划分训练集和验证集
    totalData = pd.DataFrame(saveData['final_trainData'])
    X_train, X_val, y_train, y_val = train_test_split(totalData.iloc[:, :-1], totalData.iloc[:, -1], test_size=0.25,
                                                      random_state=42)
    X_train_origin = X_train.copy()
    y_train_origin = y_train.copy()
    # 处理 additional_data
    if isinstance(additional_data, pd.DataFrame):
        # 如果已经是 DataFrame，直接拷贝并重置列名
        additional_df = additional_data.copy()
        additional_df.columns = totalData.columns
    else:
        # 否则按原逻辑从 array/list 构造
        additional_df = pd.DataFrame(additional_data, columns=totalData.columns)
    # 生成的数据
    additional_X = additional_df.iloc[:, :-1]
    additional_y = additional_df.iloc[:, -1]

    X_train_expand = pd.concat([X_train, additional_X], ignore_index=True)
    y_train_expand = pd.concat([y_train, additional_y], ignore_index=True)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_expand, y_train_expand)

    y_pred = model.predict(X_val)

    # 计算F1分数
    # expand_f1 = f1_score(y_val, y_pred)
    # 计算错误率
    expand_error = 1 - accuracy_score(y_val, y_pred)

    model_origin = DecisionTreeClassifier(random_state=42)
    model_origin.fit(X_train_origin, y_train_origin)

    y_pred_origin = model_origin.predict(X_val)

    # 计算F1分数
    # origin_f1 = f1_score(y_val, y_pred_origin)
    # 计算错误率
    origin_error = 1 - accuracy_score(y_val, y_pred_origin)

    # print(f'Error Rate: {expand_error}')
    # print(f'origin Error Rate: {origin_error}')
    # if origin_error != 0:
    #     err_percent = 1 - expand_error / origin_error
    # else:
    #     err_percent = 0
    # print(f"error rate improve: {100 * err_percent}")
    # print(f'F1 Score: {expand_f1}')
    # print(f'origin F1 Score: {origin_f1}')
    # f1_gain = expand_f1 - origin_f1
    # print(f"f1 rate improve: {f1_gain}")
    return origin_error - expand_error

# def evaluate_greedy_model(path, model_id, additional_data):
#     # 接受所有的shared model data,采用贪心策略，选择合适子集
#
    
    

# 使用示例
# path = 'dataset/clf_num/bank-marketing.csv'  # 替换为你的CSV文件路径
# evaluate(csv_file_path=path, additional_data=[[]])
