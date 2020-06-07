# 简单线性回归（梯度下降法）
import numpy as np


class GradientDescent:
    # 定义初始化方法，传入数据，以及初始权重，偏执，学习率，迭代次数
    def __init__(self, x, y, initial_w, initial_b, l_rate, iter):
        self.m, self.d = x.shape
        self.x = x
        self.y = y
        self.initial_w = initial_w
        self.initial_b = initial_b
        self.l_rate = l_rate
        self.iter = iter

    # 定义sigma求和函数
    def sigma(self, data):
        num = len(data)
        sum = 0
        for i in range(num):
            sum += data[i]
        return sum

    # 定义损失函数
    def compute_cost(self, w, b):
        # w, b = self.grad_desc()
        total_cost = self.sigma((self.y - w * self.x - b) ** 2)
        return total_cost / self.m

    # 定义拟合函数
    def fit(self):
        w = self.initial_w
        b = self.initial_b
        for i in range(self.iter):
            w, b = self.step_grad_desc(w, b)
        return w, b

    # 定义梯度下降函数
    def grad_desc(self):
        w = self.initial_w
        b = self.initial_b
        # 定义一个list保存所有的损失函数值，用来显示下降的过程
        cost_list = []

        for i in range(self.iter):
            cost_list.append(self.compute_cost(w, b))
            w, b = self.step_grad_desc(w, b)
        return w, b, cost_list

    # 定义每一步梯度下降函数
    def step_grad_desc(self, current_w, current_b):
        # 计算梯度
        grad_w = self.sigma((current_w * self.x + current_b - self.y) * self.x) / self.m
        grad_b = self.sigma(current_w * self.x + current_b - self.y) / self.m
        # 梯度下降，更新当前的w和b
        updated_w = current_w - self.l_rate * grad_w
        updated_b = current_b - self.l_rate * grad_b
        return updated_w, updated_b

    # 定义预测函数
    def predict(self, test_x):
        w, b = self.fit()
        return w * test_x + b


if __name__ == '__main__':
    # 导入绘图库
    import matplotlib.pyplot as plt

    # 导入训练数据
    points = np.genfromtxt('data.csv', delimiter=',')
    x = points[:, 0].reshape(100, 1)
    y = points[:, 1].reshape(100, 1)

    # 绘制训练数据集的散点图
    ax1 = plt.subplot(2, 1, 1)
    ax1.scatter(x, y)

    # 实例化梯度下降法
    gd = GradientDescent(x, y, 0, 0, 0.0001, 100)

    # 求解权重、偏执以及每次迭代的损失函数值
    w, b, cost_list = gd.grad_desc()
    print("w is {}, b is {}".format(w, b))

    # 绘制拟合曲线
    pred_y = w * x + b
    ax1.plot(x, pred_y, c='r')
    # 给出测试点进行预测并显示在图像上
    test_x = 60
    test_y = gd.predict(test_x)
    print(test_y)
    ax1.scatter(test_x, test_y, c='y')

    # 画出梯度下降趋势
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(cost_list)
    plt.show()
