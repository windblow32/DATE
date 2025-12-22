import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载示例数据集
iris = load_iris()
X, y = iris.data, iris.target

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 选择一个样本进行预测
sample_index = 0
sample = X_test[sample_index].reshape(1, -1)
prediction = clf.predict(sample)
print(f"Predicted class for sample {sample_index}: {prediction[0]}")

# 获取决策路径
node_indicator = clf.decision_path(sample)
leave_id = clf.apply(sample)

# 解析决策路径
feature = clf.tree_.feature
threshold = clf.tree_.threshold
node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

print("\nDecision path for sample:\n")
for node_id in node_index:
    if leave_id[0] == node_id:
        print(f"Leaf node {node_id} reached, prediction: {prediction[0]}")
    else:
        threshold_sign = "<=" if sample[0, feature[node_id]] <= threshold[node_id] else ">"
        print(f"Node {node_id}: (Feature {feature[node_id]} {threshold_sign} {threshold[node_id]})")

