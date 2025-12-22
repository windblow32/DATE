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

'''
由后向前贪心
'''

# 尝试score 取top，score和rule结合进行采样，查看只凭借score是不是无法得到有效/最优，反思验证集与测试集是否满足同分布
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
model_id = 0

# 加载原始数据
saveData = load_model(path, model_id)
final_trainData = pd.DataFrame(saveData['final_trainData'])
final_testData = pd.DataFrame(saveData['final_testData'])
X_test = final_testData.iloc[:, :-1]
y_test = final_testData.iloc[:, -1]

# 初始化基础模型（只使用原始训练数据）
X_train_origin = final_trainData.iloc[:, :-1]
y_train_origin = final_trainData.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X_train_origin, y_train_origin, test_size=0.25, random_state=42)
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_val)
base_accuracy = accuracy_score(y_val, y_pred_base)

print(f"Base model accuracy: {base_accuracy:.4f}")

# 读取PKL文件
with open(f"generated/0825_{path}_data_{model_id}.pkl", "rb") as f:
    data_objects = pickle.load(f)
    print(f"Loaded {len(data_objects)} data objects for model {model_id}")
# ==== 修改结束 ====

# 初始化贪心策略变量
best_accuracy = base_accuracy
selected_objects = []  # 存储被选中的数据对象
current_X = X_train.copy()  # 从原始训练数据开始
current_Y = y_train.copy()  # 从原始训练数据开始
scoreList = []
selected_index = []

obj_length = len(data_objects)
# 反转data_objects的顺序
data_objects = data_objects[::-1]

# 验证贪心选择性
# 只处理 topk 的数据对象
for idx, data_obj in tqdm(enumerate(data_objects), total=len(data_objects), desc="Processing top k data objects"):
    # 后向贪心，先加入后面的example

    print(f"index {len(data_objects) - 1 - idx}")
    scoreList.append(data_obj['score'])
    # 将数据对象转换为DataFrame
    data_array = data_obj['data_list']
    obj_df = pd.DataFrame(data_array, columns=final_trainData.columns)

    # 创建临时训练集（当前数据+新数据）
    X_temp = pd.concat([current_X, obj_df.iloc[:, :-1]], ignore_index=True)
    y_temp = pd.concat([current_Y, obj_df.iloc[:, -1]], ignore_index=True)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_temp, y_temp)
    # 用验证集，防止数据泄漏
    y_pred = model.predict(X_val)
    current_accuracy = accuracy_score(y_val, y_pred)

    # 如果准确率提升，保留这个数据对象
    if current_accuracy > best_accuracy:
        print(f"greedy gain(迭代的增加): {current_accuracy - best_accuracy}")
        best_accuracy = current_accuracy
        selected_index.append(idx)
        current_X = X_temp.copy()  # 更新当前数据
        current_Y = y_temp.copy()  # 更新当前数据
        selected_objects.append(data_obj)
        print(f"Added object: Accuracy improved to {best_accuracy:.4f}")
        print(f"  Path: {data_obj['path']}")
        print(f"  Origin Score: {data_obj['score']:.4f}")
        print(f"  Samples: {len(obj_df)}")
        print(f"Current error: {1 - current_accuracy:.4f}\n")
    else:
        print(f"\nSkipped object: Accuracy {current_accuracy:.4f} (Current best: {best_accuracy:.4f})")
        print(f"  Path: {data_obj['path']}\n")

print("========validation error=========")
print(f"Final error rate: {1 - best_accuracy:.4f}")
print(f"original validation: {1 - base_accuracy}")
print(selected_index)

# 最终使用所有被选中的数据对象构建增强数据集
# 注意不是直接使用current_df（需要实验查看哪个效果好，有可能X_train + select_objects效果不错）
selected_df = pd.DataFrame()
for obj in selected_objects:
    selected_df = pd.concat([selected_df, pd.DataFrame(obj['data_list'], columns=final_trainData.columns)],
                            ignore_index=True)
# concat selected_df with final_trainData
# X_train_expand = pd.concat([X_train.copy(), selected_df.iloc[:, :-1]], ignore_index=True)
# y_train_expand = pd.concat([y_train.copy(), selected_df.iloc[:, -1]], ignore_index=True)
expand = pd.concat([selected_df, final_trainData.copy()], ignore_index=True)
X_train_expand = expand.iloc[:, :-1]
y_train_expand = expand.iloc[:, -1]

# 训练最终模型
final_model = DecisionTreeClassifier(random_state=42)
final_model.fit(X_train_expand, y_train_expand)

# 评估最终模型
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
final_error = 1 - final_accuracy

# 原始模型评估
model_origin = DecisionTreeClassifier(random_state=42)
model_origin.fit(X_train_origin.copy(), y_train_origin.copy())
y_pred_origin = model_origin.predict(X_test)
origin_accuracy = accuracy_score(y_test, y_pred_origin)
origin_error = 1 - origin_accuracy

# 打印结果
print("\n" + "=" * 50)
print(f"Base model accuracy: {base_accuracy:.4f}")
print(f"Final model accuracy: {final_accuracy:.4f}")
print(f"Accuracy improvement: {final_accuracy - base_accuracy:.4f}")
print(f"Selected data objects: {len(selected_objects)}/{len(data_objects)}")

print(f'\nError Rate: {final_error:.4f}')
print(f'Origin Error Rate: {origin_error:.4f}')
if origin_error != 0:
    err_percent = 1 - final_error / origin_error
else:
    err_percent = 0
print(f"Error rate improvement: {100 * err_percent:.2f}%")
