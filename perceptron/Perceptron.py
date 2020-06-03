import numpy as np


class Perceptron:
    # 初始化模型参数
    def __init__(self, train_data, train_target, b, l_rate=1):
        self.train_data = train_data
        self.train_target = train_target
        self.w = np.zeros((1, np.shape(self.train_data)[1]))
        self.b = b
        self.l_rate = l_rate

    # 定义符号函数
    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        return x

    # 定义求解权重矩阵和偏执函数
    def fit(self):
        while self.compute_loss(self.train_data, self.train_target, self.w, self.b) != 0:
            # print(self.compute_loss())
            for i in range(len(self.train_data)):
                x = self.train_data[i]
                y = self.train_target[i]
                # print(x, y, self.w, self.b, self.sign(np.dot(self.w, x) + self.b) == y)
                if self.sign(np.dot(self.w, x) + self.b) != y:
                    self.w += self.l_rate * np.dot(y, x)
                    self.b += self.l_rate * y
                # time.sleep(1)
        return self.w, self.b

    # 定义损失函数
    def compute_loss(self, data, target, w, b):
        total_loss = 0
        for i in range(len(data)):
            if self.sign(np.dot(w, data[i]) + b) != target[i]:
                total_loss -= target[i] * (np.dot(w, data[i]) + b)
                # print(total_loss)
        return total_loss

    # 定义预测函数
    def predict(self, test_data):
        w, b = self.fit()
        return self.sign(np.dot(w, test_data) + b)


if __name__ == "__main__":

    # test1
    # X = np.array([
    #     [3, 3],
    #     [4, 3],
    #     [1, 1],
    #     [3,0],
    #     [4,0]
    # ])
    # Y = np.array([1, 1, -1,-1,-1])
    # w = np.ones((1, np.shape(X)[1]))
    # b = 1
    # p = Perceptron(X, Y, b)
    # result = p.fit()
    # y=p.predict(np.array([[3],[4]]))
    # print(result)
    # print(y)

    # test2
    from sklearn.datasets import load_iris

    data, target = load_iris(return_X_y=100)
    data = data[:100]
    # print(data)
    target = target[:100].reshape((100, 1))
    target[:50] = -1
    # print(target)
    train_set = np.hstack((data, target))
    np.random.shuffle(train_set)
    # print(train_set)
    train_data = train_set[:80, :4]
    train_targert = train_set[:80, 4]
    p = Perceptron(train_data, train_targert, 1, 0.01)
    test_set = train_set[80:, :4]
    value = train_set[80:, 4].reshape((20, 1))
    print(train_set[80:, 4])
    num = len(test_set)
    count = 0
    for i in range(num):
        y_pred = p.predict(test_set[i])
        print(y_pred)
        if y_pred == value[i]:
            count += 1
    print(count / 20)
