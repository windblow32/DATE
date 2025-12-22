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
from support import compute_rules_fractional_support


def load_model(dataset, model_id):
    """Load a model from disk using pickle."""
    filename = f"model/{dataset}_model_{model_id}.pkl"
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_id} not found")
        return None


class AdaptiveBanditSelector:
    def __init__(self, data_objects, base_accuracy, sim_func,
                 alpha=0.9, beta=0, gamma=0.1, ucb_c=2.0):
        """
        data_objects: list of dict，每个包含 'data_list','score','path'
        base_accuracy: 原始模型在验证集上的 accuracy
        sim_func: callable(path_i, path_j)->float
        alpha,beta,gamma: 奖励／偏置权重
        ucb_c: UCB中的探索因子
        """
        self.objs = data_objects
        self.N = len(data_objects)
        self.base_acc = base_accuracy
        self.sim = sim_func
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ucb_c = ucb_c

        # 位置偏置: 越往后 index 越大
        self.pos_bias = np.arange(self.N) / float(max(self.N - 1, 1))

        # 归一化基础 reward = alpha*(base_acc + score)
        raw = [self.alpha * (self.base_acc + obj['score'])
               for obj in self.objs]
        raw = np.array(raw).reshape(-1, 1)
        self.norm_reward = MinMaxScaler().fit_transform(raw).flatten()

        # 记录已选臂
        self.selected = []
        # 记录各臂被测试次数（用于 UCB 的探索项）
        self.counts = np.zeros(self.N, dtype=int)
        # 记录各臂的平均奖励
        self.values = np.zeros(self.N, dtype=float)

        # 初始化一次性得分
        self._rescore_all()

    def _rescore_all(self):
        """重新计算所有臂的 UCB 基础值，不包含探索项"""
        support_list = []
        for i, obj in enumerate(self.objs):
            # alpha weight
            base = self.alpha * self.norm_reward[i] + self.beta * self.pos_bias[i]
            ruleList = []
            ruleList.append(obj['path'])
            support_df = compute_rules_fractional_support(final_testData.iloc[:, :-1], ruleList)
            support_ratio = support_df['support_ratio'].values[0]
            support_list.append(support_ratio)
            # if not self.selected:
            #     sim_part = 0.0
            # else:
            #     sims = [get_rule_similarity(obj['path'], json_path, self.objs[j]['path'], model_id)
            #             for j in self.selected]
            #     sim_part = max(sims)
            self.values[i] = base + self.gamma * support_ratio
            if i == 31:
                a = 1
        # 查看support情况
        print(f"support list: {support_list}")
        print(f"value list: {self.values}")

    def select_arm(self, tested_arms):
        """
        从尚未加入训练集的 arm 中选择下一个要测试的臂，
        并跳过本状态下已测试过的臂。
        tested_arms: set, 在当前 S 下已经测试过的臂索引
        """
        candidates = [i for i in range(self.N)
                      if i not in self.selected and i not in tested_arms]
        if not candidates:
            return None
        total_tests = np.sum(self.counts) + 1
        ucb_scores = []
        for i in candidates:
            if self.counts[i] == 0:
                # 未测试过，优先探索
                score = self.values[i] + self.ucb_c * math.sqrt(math.log(total_tests))
            else:
                # UCB = 平均奖励 + 探索项
                expl = self.ucb_c * math.sqrt(math.log(total_tests) / self.counts[i])
                score = self.values[i] + expl
            ucb_scores.append((score, i))
        # 选 UCB 分最高的臂
        return max(ucb_scores)[1]

    def update(self, arm, reward):
        """用新的 reward 更新 counts 和 average values"""
        self.counts[arm] += 1
        n = self.counts[arm]
        # online update 均值
        self.values[arm] += (reward - self.values[arm]) / n

    def pick_until_no_gain(self, X_train_df, y_train_df, X_val, y_val, max_rounds=100):
        """
        1. baseline 评估
        2. 在当前训练集 S 下，反复
           a) select_arm(tested_arms)
           b) test reward = accuracy(S ∪ arm) 
           c) update(arm, reward)
           d) 如果 reward>0，则加入 S，清空 tested_arms
           e) 否则 tested_arms.add(arm)
           f) 当 tested_arms == all 未选臂，停止
        """

        # 1. baseline
        def eval_df(X, y):
            m = DecisionTreeClassifier(random_state=0).fit(X, y)
            return accuracy_score(y_val, m.predict(X_val))

        best_acc = eval_df(X_train_df, y_train_df)
        print("Baseline acc:", best_acc)
        tested_arms = set()
        selected_arms = []

        for _ in range(max_rounds):
            arm = self.select_arm(tested_arms)
            if arm is None:
                break

            # 测试 reward
            # 准备加入的子集
            obj_df = pd.DataFrame(self.objs[arm]['data_list'], columns=final_trainData.columns)

            # 创建临时训练集（当前数据+新数据）
            # bank-marketing modelid=2
            data = final_trainData.copy()
            X_temp = pd.concat([data.iloc[:, :-1], obj_df.iloc[:, :-1]], ignore_index=True)
            y_temp = pd.concat([data.iloc[:, -1], obj_df.iloc[:, -1]], ignore_index=True)

            acc = eval_df(X_temp, y_temp)
            reward = acc - best_acc

            # 更新 bandit
            self.update(arm, reward)

            print(f"Test arm {arm}: acc={acc:.4f}, gain={reward:.4f}")

            if reward > 0:
                # 接受 arm，更新 S, best_acc，清空 tested_arms
                X_train_df = X_temp
                y_train_df = y_temp
                best_acc = acc
                selected_arms.append(arm)
                tested_arms.clear()
                # rescore 新的 values 中 sim 部分
                self.selected.append(arm)
                self._rescore_all()
                print(f"  => accept arm {arm}, new best_acc={best_acc:.4f}")
            else:
                tested_arms.add(arm)
                print(f"  => skip arm {arm}")

            # 如果目前所有未选臂都测过一次且无增益，停止
            remaining = set(range(self.N)) - set(self.selected)
            if tested_arms >= remaining:
                print("No further gain, stop.")
                break

        return selected_arms, best_acc


# 主程序
path = "bank-marketing"
# set target id
model_id = 4
json_path = "current_block/bank-marketing_ROU=0.2_f1-error_block.json"

# 读取PKL文件
with open(f"generated/0825_{path}_data_{model_id}.pkl", "rb") as f:
    data_objects = pickle.load(f)
    print(f"Loaded {len(data_objects)} data objects for model {model_id}")

# 只保留reflect层面

data_objects = data_objects[:]
print(len(data_objects))

# 加载原始数据
saveData = load_model(path, model_id)
df = pd.read_csv("dataset/clf_num/" + path + ".csv", nrows=0)
# 获取DataFrame的列名，即表头
header = df.columns.tolist()
final_trainData = pd.DataFrame(saveData['final_trainData'], columns=header)
final_testData = pd.DataFrame(saveData['final_testData'], columns=header)
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


selector = AdaptiveBanditSelector(
    data_objects=data_objects,
    base_accuracy=base_accuracy,
    sim_func=compute_rules_fractional_support,
    alpha=0.9,
    beta=0,
    gamma=0.1,
    ucb_c=2.0
)

selected_indices, final_acc = selector.pick_until_no_gain(
    X_train_df=X_train.copy(),
    y_train_df=y_train.copy(),
    X_val=X_val, y_val=y_val,
    max_rounds=100
)
selected_df = pd.DataFrame()
for idx in selected_indices:
    print(idx)
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

print(f'\nError Rate: {final_error:.4f}')
print(f'Origin Error Rate: {origin_error:.4f}')
if origin_error != 0:
    err_percent = 1 - final_error / origin_error
else:
    err_percent = 0
print(f"Error rate improvement: {100 * err_percent:.2f}%")
