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


class MultiArmedBanditSelector:
    def __init__(self, data_objects, model_id, json_path,
                 ucb_c=2.0, beta=0.2, alpha=0.3):
        """
        data_objects: 列表，每项有 'data_list','score','path' 等
        ucb_c: UCB 算法的探索参数
        beta: 位置偏置强度，beta>0 时越后面的臂会被额外加分
        """
        self.data_objects = data_objects
        self.ucb_c = ucb_c
        self.beta = beta
        self.model_id = model_id
        self.json_path = json_path
        self.alpha = alpha

        N = len(data_objects)
        # 位置偏置：id 越大 bias 越大
        self.pos_bias = np.arange(N) / float(max(N - 1, 1))

        # 计数、均值
        self.counts = np.zeros(N, dtype=int)
        self.values = np.zeros(N, dtype=float)

        # 预先算一次初始 reward（可选），并归一化
        base_rewards = []
        for i, obj in enumerate(data_objects):
            sim = get_rule_similarity(obj['path'], json_path, model_id)
            # 这里只示例：score + sim 组合
            r = self.alpha * obj['score'] * 10 + (1 - self.alpha - self.beta) * sim
            base_rewards.append(r)
        base_rewards = np.array(base_rewards).reshape(-1, 1)
        self.norm_rewards = MinMaxScaler().fit_transform(base_rewards).flatten()

        # 初始给未被选过的臂设置一个初始平均值
        # 这样在 select_arm 时就无需「inf」判定
        self.values[:] = self.norm_rewards + self.beta * self.pos_bias

    def select_arm(self):
        t = np.sum(self.counts) + 1
        ucb_values = np.zeros_like(self.values)
        for i in range(len(self.data_objects)):
            if self.counts[i] == 0:
                # 从未被选：用初始 value + 一个大探索项
                ucb_values[i] = self.values[i] + self.ucb_c * math.sqrt(math.log(t))
            else:
                exploration = self.ucb_c * math.sqrt(math.log(t) / self.counts[i])
                ucb_values[i] = self.values[i] + exploration + self.beta * self.pos_bias[i]

        return int(np.argmax(ucb_values))

    def update(self, chosen_arm, reward):
        # 累计次数
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        # 更新平均值
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n


# 主程序
path = "credit"
model_id = 1
json_path = "current_block/"+path+"_ROU=0.2_f1-error_block.json"

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

# 初始化多臂老虎机选择器
selector = MultiArmedBanditSelector(data_objects, model_id, json_path, ucb_c=2.0, beta=0.2, alpha=-0.2)

# 初始化贪心策略变量
best_accuracy = base_accuracy
current_df = final_trainData.copy()  # 从原始训练数据开始
selected_objects = []  # 存储被选中的数据对象
selected_indices = []  # 存储被选中的数据对象索引
t = 1  # 时间步长计数器

# 贪心选择直到准确率不再提升或达到最大迭代次数
max_iterations = max(50, len(data_objects) * 2)  # 最大迭代次数
accuracy_history = [base_accuracy]

for iteration in range(max_iterations):
    # 使用多臂老虎机选择臂
    arm_index = selector.select_arm()
    data_obj = data_objects[arm_index]
    print(f"choose index {arm_index}")

    # 将数据对象转换为DataFrame
    data_array = data_obj['data_list']
    obj_df = pd.DataFrame(data_array, columns=final_trainData.columns)

    # 创建临时训练集（当前数据+新数据）
    # temp_df = pd.concat([current_df, obj_df], ignore_index=True)

    # 创建临时训练集（当前数据+新数据）
    X_temp = pd.concat([X_train.copy(), obj_df.iloc[:, :-1]], ignore_index=True)
    y_temp = pd.concat([y_train.copy(), obj_df.iloc[:, -1]], ignore_index=True)

    model = DecisionTreeClassifier(random_state=42)  # 减少树的数量以提高速度
    model.fit(X_temp, y_temp)

    y_pred = model.predict(X_val)
    current_accuracy = accuracy_score(y_val, y_pred)

    # 计算当前奖励（准确率提升）
    score = current_accuracy
    sim = get_rule_similarity(data_obj['path'], json_path, model_id)
    reward = selector.alpha * score*10 + (1 - selector.alpha - selector.beta) * (sim)

    # 更新选择器
    selector.update(arm_index, reward)

    # 如果准确率提升，保留这个数据对象
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        # 每个独立测量
        # current_df = temp_df.copy()  # 更新当前数据
        selected_objects.append(data_obj)
        selected_indices.append(arm_index)
        print(f"Iteration {iteration}: Added object {arm_index}")
        print(f"  Path: {data_obj['path']}")
        print(f"  Score: {data_obj['score']:.4f}")
        print(f"  Samples: {len(obj_df)}")
        print(f"  Accuracy improved to {best_accuracy:.4f} (+{current_accuracy - accuracy_history[-1]:.4f})\n")
    else:
        print(f"Iteration {iteration}: Skipped object {arm_index}")
        print(f"  Path: {data_obj['path']}")
        print(f"  Score: {data_obj['score']:.4f}")
        print(f"  Accuracy: {current_accuracy:.4f} (Current best: {best_accuracy:.4f})\n")

    # 记录准确率历史
    accuracy_history.append(current_accuracy)
    t += 1

selected_df = pd.DataFrame()
for i in selected_indices:
    # 如果效果不理想可以尝试添加这个，约束一下只保留后面新生成的
    if i < 26:
        continue
    selected_df = pd.concat([selected_df, pd.DataFrame(data_objects[i]['data_list'], columns=final_trainData.columns)],
                            ignore_index=True)
# for obj in selected_objects:
#     selected_df = pd.concat([selected_df, pd.DataFrame(obj['data_list'], columns=final_trainData.columns)],
#                             ignore_index=True)
# concat selected_df with final_trainData
# 最终使用所有被选中的数据对象构建增强数据集
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
print(f"Selected data objects: {len(selected_objects)}/{len(data_objects)}")
print(f"Final training size: {len(current_df)}")

print(f'\nError Rate: {final_error:.4f}')
print(f'Origin Error Rate: {origin_error:.4f}')
if origin_error != 0:
    err_percent = 1 - final_error / origin_error
else:
    err_percent = 0
print(f"Error rate improvement: {100 * err_percent:.2f}%")
