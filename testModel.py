import pickle
from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ModelShare_with_DSR_final import Rule
from ModelShare_with_DSR_final import Predicate

from ModelShare_with_DSR_final import Model


def load_test(model_id):
    save_path = f"model/testset/{datasetName}_subset_{model_id}.csv"
    data = pd.read_csv(save_path)
    return data


def evaluate_singleModel(model_id):
    print(f"model id: {model_id}")
    saveData = load_model(datasetName, model_id)
    final_trainData = pd.DataFrame(saveData['final_trainData'])
    final_testData = pd.DataFrame(saveData['final_testData'])
    # final_testData = load_test(model_id)

    # Concatenate them row-wise

    X_test = final_testData.iloc[:, :-1]
    y_test = final_testData.iloc[:, -1]

    X_train_origin = final_trainData.iloc[:, :-1]
    y_train_origin = final_trainData.iloc[:, -1]
    model_origin = DecisionTreeClassifier(random_state=42)
    model_origin.fit(X_train_origin, y_train_origin)

    y_pred_origin = model_origin.predict(X_test)

    # 计算F1分数
    # origin_f1 = f1_score(y_test, y_pred_origin)
    # 计算错误率
    origin_error = 1 - accuracy_score(y_test, y_pred_origin)

    print(f'origin Error Rate: {origin_error}')
    return origin_error


def calcValidError(modelid):
    print(f"==========model {modelid}==========")
    current_offline_model = load_model(dataset=datasetName, model_id=modelid)
    train = current_offline_model.get('final_trainData', [])
    class_distribution = Counter([row[-1] for row in train])
    class_1_count = class_distribution.get(int(classLabel_1), 0)
    class_2_count = class_distribution.get(int(classLabel_2), 0)
    print(f"trainset class 1: {class_1_count}")
    print(f"trainset class 2: {class_2_count}")
    train = pd.DataFrame(train)
    X_train, X_val, y_train, y_val = train_test_split(train.iloc[:, :-1], train.iloc[:, -1], test_size=0.25,
                                                      random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    model_pred = model.predict(X_val)
    p = accuracy_score(y_val, model_pred)
    f = f1_score(y_val, model_pred)
    print(f"F1: {f}")
    print(f"accuracy: {p}")
    error_rate = 1 - p
    if error_rate == 1.0:
        print(0.0)
    else:
        print(f"error rate: {error_rate}")
    print(f"validation subset size: {X_val.shape[0]}")
    if p != 0:
        error = X_val.shape[0] * (1 - p)
    else:
        error = 0
    return error, X_val.shape[0]


def calcTestError(modelid):
    print(f"==========model {modelid}==========")
    current_offline_model = load_model(dataset=datasetName, model_id=modelid)
    train = current_offline_model.get('final_trainData', [])
    class_distribution = Counter([row[-1] for row in train])
    class_1_count = class_distribution.get(int(classLabel_1), 0)
    class_2_count = class_distribution.get(int(classLabel_2), 0)
    print(f"trainset class 1: {class_1_count}")
    print(f"trainset class 2: {class_2_count}")

    final_trainData = pd.DataFrame(current_offline_model['final_trainData'])
    final_testData = pd.DataFrame(current_offline_model['final_testData'])
    # final_testData = load_test(modelid)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(final_trainData.iloc[:, :-1], final_trainData.iloc[:, -1])
    model_pred = model.predict(final_testData.iloc[:, :-1])
    p = accuracy_score(final_testData.iloc[:, -1], model_pred)
    print(f"accuracy: {p}")
    f = f1_score(final_testData.iloc[:, -1], model_pred)
    print(f"F1: {f}")
    error_rate = 1 - p
    if error_rate == 1.0:
        print(0.0)
    else:
        print(f"error rate: {error_rate}")
    print(f"final_trainData size: {final_trainData.shape[0]}")
    if p != 0:
        error = final_testData.shape[0] * (1 - p)
    else:
        error = 0
    return error, final_testData.shape[0]


def calcWholeDataError():
    modelid = -1
    final_trainData = pd.DataFrame()
    # 加载所有modelid的traindata，加载id=-1的test， 测试我们的效果
    for id in range(9):
        train_model = load_model(dataset=datasetName, model_id=id)
        train = train_model.get('final_trainData', [])
        final_trainData = pd.concat([final_trainData, pd.DataFrame(train)], ignore_index=True)

    # 从csv中加载id=-1的test
    wholeTestPath = f"model/testset/{datasetName}_subset_{modelid}.csv"
    final_testData = pd.read_csv(wholeTestPath)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(final_trainData.iloc[:, :-1], final_trainData.iloc[:, -1])
    model_pred = model.predict(final_testData.iloc[:, :-1])
    p = accuracy_score(final_testData.iloc[:, -1], model_pred)
    print(f"accuracy: {p}")
    f = f1_score(final_testData.iloc[:, -1], model_pred)
    print(f"F1: {f}")
    error_rate = 1 - p
    if error_rate == 1.0:
        print(0.0)
    else:
        print(f"error rate: {error_rate}")
    print(f"final_trainData size: {final_trainData.shape[0]}")
    if p != 0:
        error = final_testData.shape[0] * (1 - p)
    else:
        error = 0
    return error, final_testData.shape[0]


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


def originEvaluation(name):
    path = "dataset/clf_num/" + name + ".csv"
    df = pd.read_csv(path)
    X_train, X_val, y_train, y_val = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    p = accuracy_score(y_val, y_pred)
    f = f1_score(y_val, y_pred)
    print(f"F1: {f}")
    print(f"accuracy: {p}")
    error_rate = 1 - p
    if error_rate == 1.0:
        print(0.0)
    else:
        print(f"error rate: {error_rate}")
    return error_rate


def originEvaluation_rf(name):
    path = "dataset/clf_num/" + name + ".csv"
    df = pd.read_csv(path)
    X_train, X_val, y_train, y_val = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    p = accuracy_score(y_val, y_pred)
    f = f1_score(y_val, y_pred)
    print(f"F1: {f}")
    print(f"accuracy: {p}")
    error_rate = 1 - p
    if error_rate == 1.0:
        print(0.0)
    else:
        print(f"error rate: {error_rate}")
    return error_rate


# 修改dataset
datasetName = "eye_movements"
classLabel_1 = "0"
classLabel_2 = "1"

calcValidError(0)
calcValidError(1)
# calcValidError(2)
# calcValidError(3)
# calcValidError(4)
# calcValidError(5)

print("========Test========")

error_0, shape_0 = calcTestError(0)
error_1, shape_1 = calcTestError(1)
# error_2, shape_2 = calcTestError(2)
# error_3, shape_3 = calcTestError(3)
# error_4, shape_4 = calcTestError(4)
# error_5, shape_5 = calcTestError(5)
# error_6, shape_6 = calcWholeDataError()

print("=======test size======")
print(shape_0)
print(shape_1)
# print(shape_2)
# print(shape_3)
# print(shape_4)
# print(shape_5)
# print("========Whole Data========")
# print(f"whole data size {shape_6}")
# print(f"whole data error: {error_6 / shape_6}")

print("==========score==========")
# total error rate

# error_rate = (error_0 + error_1 + error_6)/(shape_0 + shape_1 + shape_6)
# print(f"subset split total: {error_rate}")

# error_rate2 = (
#                           shape_0 * 0.2175 + shape_1 * 0 + shape_2 * 0.1951 + shape_3 * 0.1333 + shape_4 * 0.1667 + shape_5 * 0.1905) / (
#                       shape_0 + shape_1 + shape_2 + shape_3 + shape_4 + shape_5)
# print(f"MAB total: {error_rate2}")
print("==========original dataset===========")
error_rate3 = originEvaluation(datasetName)
# print(f"origin total: {error_rate3}")
