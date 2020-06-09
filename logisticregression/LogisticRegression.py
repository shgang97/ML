import numpy as np


# 逻辑斯蒂回归分类器
class LogisticRegressionClassifier:
    def __init__(self, x, y, initial_w, l_rate=0.001, iter=1000, ):
        self.m = x.shape[0]
        self.x = np.concatenate((np.ones((self.m, 1)), x), axis=1)
        self.y = y
        self.initial_w = initial_w
        self.l_rate = l_rate
        self.iter = iter

    def sign(self, data):
        m = data.shape[0]
        result = np.ones((m, 1))
        for i in range(m):
            if data[i][0] > 0.5:
                result[i][0] = 1
            else:
                result[i][0] = 0
        return result

    # 定义sigmoid函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 定义sigma求和函数
    def sigma(self, data):
        num = len(data)
        sum = 0
        for i in range(num):
            sum += data[i]
        return sum

    # 定义每一步更新权重矩阵函数
    def step_grad_desc(self, current_w):
        # 计算梯度
        grad = self.sigma(
            self.x * (self.y - np.exp(np.dot(self.x, current_w)) / (1 + np.exp(np.dot(self.x, current_w)))))
        # 更新权重矩阵并返回
        update_w = current_w + self.l_rate * grad.reshape((5, 1))
        return update_w

    # 进行迭代求解权重矩阵w
    def grad_desc(self):
        w = self.initial_w
        for i in range(self.iter):
            w = self.step_grad_desc(w)
        return w

    # 定义预测函数
    def predict(self, test_x):
        m = test_x.shape[0]
        test_x = np.concatenate((np.ones((m, 1)), test_x), axis=1)
        w = self.grad_desc()
        return self.sign(self.sigmoid(np.dot(test_x, w)))


if __name__ == '__main__':
    # 从sklearn导入鸢尾花数据集
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split  # 切分数据集为训练集和测试集
    from sklearn.metrics import accuracy_score  # 计算分类预测的准确率

    # 对数据进行预处理
    iris = load_iris()
    data_set = np.hstack((iris.data[:100], iris.target[:100].reshape((-1, 1))))
    x = data_set[:, :4]
    y = data_set[:, 4].reshape((-1, 1))
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)

    # 创建一个LogisticRegression分类器实例
    # 定义一个初始化权值矩阵
    initial_w = np.zeros((5, 1))
    lrc = LogisticRegressionClassifier(x_train, y_train, initial_w)
    w = lrc.grad_desc()
    print(w)
    # 测试
    pred_y = lrc.predict(x_test)
    scores = accuracy_score(y_test, pred_y)
    print("预测准确率为：%.2f%%" % (scores * 100))