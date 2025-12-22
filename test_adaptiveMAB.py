import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import math
from ModelShare_with_DSR_final import Rule
from ModelShare_with_DSR_final import Predicate
from ModelShare_with_DSR_final import Model
from ruleSim import get_rule_similarity


def load_model(dataset, model_id):
    """Load a model from disk using pickle."""
    filename = f"model/{dataset}_model_{model_id}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_id} not found")
        return None


class AdaptiveOneShotSelector:
    def __init__(self, data_objects, base_accuracy, sim_func,
                 alpha=0.9, beta=1.0, gamma=0.1):
        """
        data_objects: list of dicts, each has 'data_list','score','path'
        base_accuracy: float, 原始模型在验证集上的 accuracy
        sim_func: callable(path_i, path_j)->float, 自定义相似度
        alpha, beta, gamma: 各部分权重
        """
        self.objs = data_objects
        self.N = len(data_objects)
        self.base_acc = base_accuracy
        self.sim = sim_func
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # 位置偏置：后面的臂 bias 更大
        self.pos_bias = np.arange(self.N) / float(max(self.N - 1, 1))

        # 预计算并归一化每个臂的 base reward
        raw = []
        for obj in self.objs:
            # r = alpha*(base_acc + obj['score'])
            raw.append(self.alpha * (self.base_acc + obj['score']))
        raw = np.array(raw).reshape(-1, 1)
        self.norm_reward = MinMaxScaler().fit_transform(raw).flatten()

        # 记录哪些臂已经被选
        self.selected = []

        # 初始化一次性得分
        self._recompute_scores()

    def _recompute_scores(self):
        """重算所有臂的 total score"""
        scores = []
        for i, obj in enumerate(self.objs):
            # 基础部分：归一化 reward + 位置偏置
            base_part = self.norm_reward[i] + self.beta * self.pos_bias[i]
            # 相似度部分：与（已选臂和离线子集）计算的sim
            if self.selected is None:
                sim = get_rule_similarity(obj['path'], json_path, model_id, new_rules=None, new_weights=None)
            else:
                # 为计算sim准备new_rules和new_weights
                new_rules = []
                new_weights = []
                for j in self.selected:
                    new_rules.append(self.objs[j]['path'])
                    new_weights.append(len(self.objs[j]['data_list']))

                sim = get_rule_similarity(obj['path'], json_path, model_id, new_rules=new_rules,
                                          new_weights=new_weights)

            total = base_part + self.gamma * sim
            scores.append(total)
        self.scores = np.array(scores)

    def select_next(self):
        """一次性选下一个得分最高且未被选中的臂"""
        candidates = [i for i in range(self.N) if i not in self.selected]
        if not candidates:
            return None
        best = max(candidates, key=lambda i: self.scores[i])
        return best

    def pick_and_update(self):
        """选一个臂并更新其他臂得分"""
        idx = self.select_next()
        if idx is None:
            return None
        self.selected.append(idx)
        self._recompute_scores()
        return idx

    def select_until_no_gain(self,
                             X_train_df,
                             y_train_df,
                             X_val, y_val,
                             max_rounds=50):
        """
        从原始训练集 repeatedly 增加子集，直到验证 accuracy 不再提升
        返回: selected_indices, final_accuracy
        """
        y_pred = DecisionTreeClassifier(random_state=42).fit(X_train_df, y_train_df).predict(X_val)
        best_acc = accuracy_score(y_val, y_pred)
        selected_indices = []
        current_X = X_train_df.copy()
        current_Y = y_train_df.copy()

        for _ in range(max_rounds):
            idx = self.pick_and_update()
            if idx is None:
                break

            # 准备加入的子集
            obj_df = pd.DataFrame(self.objs[idx]['data_list'], columns=final_trainData.columns)

            # 创建临时训练集（当前数据+新数据）
            X_temp = pd.concat([current_X, obj_df.iloc[:, :-1]], ignore_index=True)
            y_temp = pd.concat([current_Y, obj_df.iloc[:, -1]], ignore_index=True)

            # 评估新模型
            model = DecisionTreeClassifier(random_state=0)
            model.fit(X_temp, y_temp)
            acc = accuracy_score(y_val, model.predict(X_val))

            print(f"Try arm {idx}: new acc={acc:.4f}, best acc={best_acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                current_X = X_temp
                current_Y = y_temp
                selected_indices.append(idx)
                print(f"  Accept arm {idx}, updated best acc={best_acc:.4f}")
            else:
                print(f"  Reject arm {idx}")
                continue

        return selected_indices, best_acc


# 主程序
path = "bank-marketing"
model_id = 2
json_path = "current_block/bank-marketing_ROU=0.2_f1-error_block.json"

# 加载原始数据
saveData = load_model(path, model_id)
final_trainData = pd.DataFrame(saveData['final_trainData'])
final_testData = pd.DataFrame(saveData['final_testData'])
X_test = final_testData.iloc[:, :-1]
y_test = final_testData.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(final_trainData.iloc[:, :-1], final_trainData.iloc[:, -1],
                                                  test_size=0.25, random_state=42)

# 初始化基础模型（只使用原始训练数据）
X_train_origin = final_trainData.iloc[:, :-1]
y_train_origin = final_trainData.iloc[:, -1]
base_model = DecisionTreeClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_val = base_model.predict(X_val)
base_accuracy = accuracy_score(y_val, y_pred_val)

print(f"Base model accuracy: {base_accuracy:.4f}")

# 读取PKL文件
with open(f"generated/0825_{path}_data_{model_id}.pkl", "rb") as f:
    data_objects = pickle.load(f)
    print(f"Loaded {len(data_objects)} data objects for model {model_id}")

# 只保留最后7个
data_objects = data_objects[43:]
# 初始化多臂老虎机选择器
# 让beta是0，在输入的集合中，直接去掉前40，看能否选择到合适的子集
selector = AdaptiveOneShotSelector(
    data_objects=data_objects,
    base_accuracy=base_accuracy,
    sim_func=get_rule_similarity,
    alpha=0.3,
    beta=0,
    gamma=0.2
)

# 初始化贪心策略变量
best_accuracy = base_accuracy
current_df = final_trainData.copy()  # 从原始训练数据开始
selected_indices = []  # 存储被选中的数据对象索引
t = 1  # 时间步长计数器

selected_indices, final_acc = selector.select_until_no_gain(
    X_train_df=X_train.copy(),
    y_train_df=y_train.copy(),
    X_val=X_val, y_val=y_val,
    max_rounds=20
)

selected_df = pd.DataFrame()
for idx in selected_indices:
    selected_df = pd.concat(
        [selected_df, pd.DataFrame(data_objects[idx]['data_list'], columns=final_trainData.columns)],
        ignore_index=True)
# concat selected_df with final_trainData
X_train_expand = pd.concat([X_train.copy(), selected_df.iloc[:, :-1]], ignore_index=True)
y_train_expand = pd.concat([y_train.copy(), selected_df.iloc[:, -1]], ignore_index=True)

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
print(f"Original model accuracy: {origin_accuracy:.4f}")
print(f"Final model accuracy in iteration: {final_accuracy:.4f}")
print(f"Accuracy improvement: {final_accuracy - origin_accuracy:.4f}")
print(f"Selected data objects: {len(selected_indices)}/{len(data_objects)}")
print(f"Final training size: {len(current_df)}")

print(f'\nError Rate: {final_error:.4f}')
print(f'Origin Error Rate: {origin_error:.4f}')
if origin_error != 0:
    err_percent = 1 - final_error / origin_error
else:
    err_percent = 0
print(f"Error rate improvement: {100 * err_percent:.2f}%")
