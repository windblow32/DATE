import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm  # 用于进度条显示
from ModelShare_with_DSR_final import Rule
from ModelShare_with_DSR_final import Predicate

from ModelShare_with_DSR_final import Model
from testModel import load_test

'''
用test挑选最好的子集，有信息泄露。此方法是用于查看哪些子集是最佳，设计方法尽可能接近
'''


def load_model(dataset, model_id):
    """Load a model from disk using pickle."""
    filename = f"model/{dataset}_model_{model_id}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_id} not found")
        return None


path = "credit"
model_id = 1

# 加载原始数据
saveData = load_model(path, model_id)
final_trainData = pd.DataFrame(saveData['final_trainData'])
final_testData = pd.DataFrame(saveData['final_testData'])
# final_testData = load_test(model_id)
X_test = final_testData.iloc[:, :-1]
y_test = final_testData.iloc[:, -1]

# 初始化基础模型（只使用原始训练数据）
X_train_origin = final_trainData.iloc[:, :-1]
y_train_origin = final_trainData.iloc[:, -1]
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train_origin, y_train_origin)
y_pred_base = base_model.predict(X_test)
base_accuracy = accuracy_score(y_test, y_pred_base)
X_train, X_val, y_train, y_val = train_test_split(X_train_origin, y_train_origin, test_size=0.25, random_state=42)
df = pd.concat([X_train, y_train], axis=1)

print(f"Base model accuracy: {base_accuracy:.4f}")

# 读取PKL文件
with open(f"generated/0825_{path}_data_{model_id}.pkl", "rb") as f:
    data_objects = pickle.load(f)
    print(f"Loaded {len(data_objects)} data objects for model {model_id}")

# 初始化贪心策略变量
best_accuracy = base_accuracy
selected_objects = []  # 存储被选中的数据对象
selected_indices = []  # 存储被选中的数据对象索引
current_df = final_trainData.copy()  # 从原始训练数据开始
scoreList = []
# 打乱数据对象的顺序
# np.random.shuffle(data_objects)
data_objects = data_objects[:]
# 验证贪心选择性
# 尝试添加每个数据对象，只保留那些能提升准确率的
for idx, data_obj in tqdm(enumerate(data_objects), total=len(data_objects), desc="Processing data objects"):
    scoreList.append(data_obj['score'])
    # 将数据对象转换为DataFrame
    data_array = data_obj['data_list']
    obj_df = pd.DataFrame(data_array, columns=final_trainData.columns)

    # 创建临时训练集（当前数据+新数据）
    temp_df = pd.concat([current_df.copy(), obj_df], ignore_index=True)

    # 训练模型并评估
    X_temp = temp_df.iloc[:, :-1]
    y_temp = temp_df.iloc[:, -1]

    model = DecisionTreeClassifier(random_state=42)  # 减少树的数量以提高速度
    model.fit(X_temp, y_temp)

    y_pred = model.predict(X_test)
    current_accuracy = accuracy_score(y_test, y_pred)

    # 如果准确率提升，保留这个数据对象
    if current_accuracy > best_accuracy:
        print(f"\ngreedy gain(迭代的增加): {current_accuracy - best_accuracy}")
        best_accuracy = current_accuracy
        current_df = temp_df.copy()  # 更新当前数据
        selected_objects.append(data_obj)
        selected_indices.append(idx)
        print(f"Added object {idx}: Accuracy improved to {best_accuracy:.4f}")
        print(f"  Path: {data_obj['path']}")
        print(f"  Origin Score: {data_obj['score']:.4f}")
        print(f"  Samples: {len(obj_df)}")
        print(f"current error: {1 - current_accuracy}")

print(f"final error rate: {1 - best_accuracy}")
print(f"select index: {selected_indices}")
# 最终使用所有被选中的数据对象构建增强数据集
X_train_expand = current_df.iloc[:, :-1]
y_train_expand = current_df.iloc[:, -1]

# 训练最终模型
final_model = DecisionTreeClassifier(random_state=42)
final_model.fit(X_train_expand, y_train_expand)

# 评估最终模型
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
final_error = 1 - final_accuracy

# 原始模型评估
model_origin = DecisionTreeClassifier(random_state=42)
model_origin.fit(X_train_origin, y_train_origin)
y_pred_origin = model_origin.predict(X_test)
origin_accuracy = accuracy_score(y_test, y_pred_origin)
origin_error = 1 - origin_accuracy

# 打印结果
print("\n" + "=" * 50)
print(f"Base model accuracy: {base_accuracy:.4f}")
print(f"Final model accuracy: {final_accuracy:.4f}")
print(f"Accuracy improvement: {final_accuracy - base_accuracy:.4f}")
print(f"Selected data objects: {len(selected_objects)}/{len(data_objects)}")
print(f"Final training size: {len(current_df)}")

print(f'\nError Rate: {final_error:.4f}')
print(f'Origin Error Rate: {origin_error:.4f}')
if origin_error != 0:
    err_percent = 1 - final_error / origin_error
else:
    err_percent = 0
print(f"Error rate improvement: {100 * err_percent:.2f}%")
