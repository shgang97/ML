import numpy as np


# 距离函数定义--曼哈顿距离
def manDistance(x, y):
    return np.sum(np.abs(x - y), axis=1)


# 距离函数定义--欧式距离
def eucDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


class kNN(object):
    # 定义一个初始化方法，传入训练数据数据，设置距离函数和近邻数
    def __init__(self, x_train, y_train, n_neighbors=1, dist_func=manDistance):
        self.x_train = x_train
        self.y_train = y_train
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 模型预测方法
    def predict(self, x):
        # 初始化预测分类数组
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        # 遍历输入的x数据点，取出每一个数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有训练数据计算距离
            distances = self.dist_func(self.x_train, x_test)
            # 得到的距离按照由近到远排序，取出索引值
            nn_index = np.argsort(distances)
            # 选取最近的k个点，保存它们对应的分类类别
            nn_y = self.y_train[nn_index[:self.n_neighbors]].ravel()
            # 统计类别中出现频率最高的那个，赋给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))
        return y_pred


if __name__ == '__main__':
    # 引入sklearn中iris鸢尾花数据集
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split  # 切分数据集为训练集和测试集
    from sklearn.metrics import accuracy_score  # 计算分类预测的准确率

    # 对数据进行预处理
    iris = load_iris()
    x = iris.data
    y = iris.target.reshape(-1, 1)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)

    # 定义一个knn实例
    knn = kNN(x_train, y_train, n_neighbors=3, dist_func=manDistance)
    # 传入测试数据，做预测
    y_pred = knn.predict(x_test)

    print(y_test.ravel())
    print(y_pred.ravel())

    # 求出预测准确率
    accuracy = accuracy_score(y_pred, y_test)

    print("预测准确率: ", accuracy)
