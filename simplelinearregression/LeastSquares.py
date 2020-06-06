# 简单线性回归（最小二乘法）


class LeastSuares:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num = len(x)

    def sigma(self, data):
        num = len(data)
        sum = 0
        for i in range(num):
            sum += data[i]
        return sum

    def average(self, data):
        return self.sigma(data) / len(data)

    def compute_loss(self):
        w, b = self.fit()
        total_loss = self.sigma((self.y - w * self.x - b) ** 2)
        return total_loss / self.num

    def fit(self):
        sum_xy = self.sigma(self.y * (self.x - self.average(self.x)))
        sum_x2 = self.sigma(self.x ** 2)
        w = sum_xy / (sum_x2 - (self.sigma(self.x)) ** 2 / self.num)
        b = self.sigma(self.y - w * self.x) / self.num
        return w, b

    def predict(self, x):
        w, b = self.fit()
        pred_y = w * x + b
        return pred_y


if __name__ == '__main__':
    # 导入依赖库
    import numpy as np
    import matplotlib.pyplot as plt

    # 导入数据
    points = np.genfromtxt('data.csv', delimiter=',')
    x = points[:, 0]
    y = points[:, 1]
    # 绘制散点图
    plt.scatter(x, y)
    # plt.show()
    # 使用最小二乘发进行拟合
    ls = LeastSuares(x, y)
    w, b = ls.fit()
    print("w is {}, b is {}".format(w, b))
    loss = ls.compute_loss()
    print("loss is {}".format(loss))
    # 绘制拟合曲线
    pred_y = ls.predict(x)
    plt.plot(x, pred_y, c='r')
    plt.show()
