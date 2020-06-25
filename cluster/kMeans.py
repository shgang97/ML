# 导入依赖库
import numpy as np
from scipy.spatial.distance import cdist  # 用于计算距离


class KMeans:
    # 初始化，参数n_cluster（K），迭代次数max_iter，初始质心centroids
    def __init__(self, n_cluster, max_iter=300, centroids=[]):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.centoids = np.array(centroids, dtype=np.float)

    # 训练模型，k-means聚类过程，传入原始数据
    def fit(self, data):
        # 加入没有指定初始质心，就随机选取data中的点作为初始质心
        if self.centoids.size == 0:
            self.centoids = data[np.random.randint(0, data.shape[0], self.n_cluster)]
        # 开始迭代，
        for j in range(self.max_iter):
            # 1.计算距离矩阵，得到的是一个100*6的矩阵
            distances = cdist(data, self.centoids)
            # 2.对距离按由近到远顺序，选取最近的质心点的类别，作为当前点的分类
            c_ind = np.argmin(distances, axis=1)
            # 3.对每一类数据进行均值计算，更新质心点坐标
            for i in range(self.n_cluster):
                # 排除掉没有出现在c_ind里的类别
                if i in c_ind:
                    # 选出所有类别是i的点，取data里面坐标的均值，更新第i个质心
                    self.centoids[i] = np.mean(data[c_ind == i], axis=0)

    # 实现预测方法
    def predict(self, samples):
        # 先计算距离矩阵，然后选取距离最近的那个质心的类别
        distances = cdist(samples, self.centoids)
        c_ind = np.argmin(distances, axis=1)
        return c_ind


# 进行测试
if __name__ == '__main__':
    # 定义一个函数，绘制数据集和质心的散点图
    def plotKMeans(x, y, centroids, subplot, title):
        # 分配子图
        plt.subplot(subplot)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        # 画出质心
        try:
            plt.scatter(centroids[:, 0], centroids[:, 1], c=np.array(range(6)), s=100)
            plt.title(title)
        except IndexError:
            print('没有初始化质心')


    import matplotlib.pyplot as plt
    # 导入数据集
    from sklearn.datasets.samples_generator import make_blobs

    x, y = make_blobs(n_samples=100, centers=6, random_state=1234, cluster_std=0.5)
    kmeans = KMeans(6)
    plt.figure(figsize=(16, 6))
    plotKMeans(x, y, kmeans.centoids, 121, 'Initial State')
    kmeans.fit(x)
    plotKMeans(x, y, kmeans.centoids, 122, 'Final State')
    plt.show()
