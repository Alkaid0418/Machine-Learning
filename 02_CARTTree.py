import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# 二叉树的节点
class Node():
    def __init__(self, best_feature_ID, best_split_value):
        self.feature_id = best_feature_ID      #非叶节点：用于分裂的特征编号
        self.split_value = best_split_value    #非叶节点：分裂值
        self.left_child = None
        self.right_child = None
        self.if_leaf = False                   #是否是叶节点
        self.mean_y = None                     #叶结点的预测值

    def set_left_chilren(self, a_node):
        self.left_child = a_node

    def set_right_chilren(self, a_node):
        self.right_child = a_node

    def get_left_chilren(self):
        return self.left_child

    def get_right_chilren(self):
        return self.right_child

    #把节点标记为叶节点
    def set_as_leaf(self):
        self.if_leaf = True


# 决策树管理类
class Tree4Decision():
    def __init__(self, a_node):
        self.root = a_node


# CART回归树核心逻辑
class CARTRegression():
    def __init__(self):
        self.feature_num = None     #特征维度数
        self.tree = None
        self.tree_size = 0  #记录节点数量
        self.max_depth = 8  #递归建树最大深度，防止过拟合

    #模型训练入口
    def fit(self, X, Y):
        self.feature_num = len(X[0])
        #构造样本索引列表
        ID_list = [i for i in range(len(X))]
        #创建虚拟根节点，不参与决策，真正的树从root.left_child开始
        self.tree = Tree4Decision(Node(-1, -1))
        # 开始递归建树，第一个节点作为根节点的左子节点
        self.build_tree(X, Y, ID_list, self.tree.root, True, 0)

    def build_tree(self, X, Y, ID_list, parent_node, if_left_child, depth):
        if len(ID_list) <= 2 or depth > self.max_depth:  #递归的终止条件
            leaf_node = Node(-1, -1)
            leaf_node.set_as_leaf()
            leaf_node.mean_y = np.mean([Y[i] for i in ID_list]) if len(ID_list) > 0 else 0
            if if_left_child:
                parent_node.set_left_chilren(leaf_node)
            else:
                parent_node.set_right_chilren(leaf_node)
            return

        best_feature_ID, best_split_value = 0, 0
        best_mse = -1

        # 寻找最优分裂点
        for feature_id in range(self.feature_num):
            # 获取该特征下所有唯一值作为潜在分裂点
            unique_values = np.unique(X[ID_list, feature_id])
            for val in unique_values:
                #对每一个候选分裂值计算MSE
                this_mse = self.get_mse_by_val(X, Y, ID_list, val, feature_id)
                if best_mse == -1 or this_mse < best_mse:
                    best_mse = this_mse
                    best_feature_ID = feature_id
                    best_split_value = val

        self.tree_size += 1
        #创建当前节点并挂到父节点
        this_node = Node(best_feature_ID, best_split_value)
        if if_left_child:
            parent_node.set_left_chilren(this_node)
        else:
            parent_node.set_right_chilren(this_node)

        # 执行分裂
        left_ID = [i for i in ID_list if X[i, best_feature_ID] <= best_split_value]
        right_ID = [i for i in ID_list if X[i, best_feature_ID] > best_split_value]

        if not left_ID or not right_ID:  # 如果无法继续分裂
            this_node.set_as_leaf()
            this_node.mean_y = np.mean([Y[i] for i in ID_list])
            return

        #递归构建左右子树
        self.build_tree(X, Y, left_ID, this_node, True, depth + 1)
        self.build_tree(X, Y, right_ID, this_node, False, depth + 1)

    def get_mse_by_val(self, X, Y, ID_list, split_val, feature_id):
        left_y = [Y[i] for i in ID_list if X[i, feature_id] <= split_val]
        right_y = [Y[i] for i in ID_list if X[i, feature_id] > split_val]

        mse = 0
        if left_y:
            mse += np.sum((np.array(left_y) - np.mean(left_y)) ** 2)
        if right_y:
            mse += np.sum((np.array(right_y) - np.mean(right_y)) ** 2)
        return mse

    def predict(self, X):
        pred_list = []
        for i in range(len(X)):
            # 从根节点的左子节点开始遍历
            pred = self.search_in_tree(X[i], self.tree.root.left_child)
            pred_list.append(pred)
        return pred_list

    def search_in_tree(self, features, current_node):
        while not current_node.if_leaf:
            if features[current_node.feature_id] <= current_node.split_value:
                current_node = current_node.get_left_chilren()
            else:
                current_node = current_node.get_right_chilren()
            if current_node is None: return -1
        return current_node.mean_y


# 数据准备与运行
if __name__ == '__main__':
    # 1. 加载数据
    data = pd.read_excel("ENB2012_data.xlsx")
    data = data.sample(frac=1, random_state=42)  # 打乱数据
    data_val = data.values
    X, Y = data_val[:, 0:-2], data_val[:, -2]

    # 2. 划分数据集（前 600 训练，剩余测试）
    train_x, train_y = X[:600, :], Y[:600]
    test_x, test_y = X[600:, :], Y[600:]

    # 3. 训练模型
    model = CARTRegression()
    model.fit(train_x, train_y)
    # 4. 在测试集上进行预测

    prediction = model.predict(test_x)

    # 5. 计算并显示误差
    mse_eval = np.mean((test_y - np.array(prediction)) ** 2)
    print(f"测试集均方误差 (MSE): {mse_eval:.4f}")

    # 可视化
    plt.figure(figsize=(10, 5))

    # 分别绘制真实值和预测值的散点图
    x_indices = range(len(test_y))
    plt.scatter(x_indices, test_y, label="True Value", color='blue', alpha=0.7, s=50)
    plt.scatter(x_indices, prediction, label="Predicted Value", color='red', s=50, marker='x')

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
