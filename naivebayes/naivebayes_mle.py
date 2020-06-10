import numpy as np


# 定义返回最大值索引函数
def agrmax(list):
    result = []
    for c, prob in list:
        result.append(c)
    return result


# 定义指示函数
def indicate(y, c):
    if y == c:
        return 1
    else:
        return 0


# 定义基于MLE的bayes分类器
class NaiveBayesClassifier:
    def __init__(self, X, Y):
        self.N, self.J = X.shape
        self.X = X
        self.Y = Y
        self.xvalues = []
        self.row = 0
        for j in range(self.J):
            x = np.unique(X[:, j])
            self.row += x.size
            self.xvalues.append(x)
        self.yvalues = np.unique(Y)
        self.K = self.yvalues.size

    # 定义计算输出空间上不同随机变量的频数
    def calc_freq(self):
        frequency = {}
        for c in self.yvalues:
            count = 0
            for i in range(self.N):
                count += indicate(self.Y[i], c)
            frequency[c] = count
        return frequency

    # 定义求解先验概率函数
    def calc_prior_prob(self):
        frequency = self.calc_freq()
        return {k: (v / self.N) for k, v in frequency.items()}

    # 定义求解条件概率函数
    def calc_cond_prob(self):
        frequency = self.calc_freq()
        cond_prob = {}
        for c in self.yvalues:
            freq = frequency[c]
            x_dim = {}
            for j in range(self.J):
                x_val = {}
                for a in self.xvalues[j]:
                    count = 0
                    for n in range(self.N):
                        # print(self.X[n][j], a)
                        count += 1 if (indicate(self.X[n][j], a) and indicate(self.Y[n], c)) else 0
                    x_val[a] = count / freq
                x_dim[j] = x_val
            cond_prob[c] = x_dim
        return cond_prob

    # 定义求解"后验概率"函数
    def calc_post_prob(self, test_X):
        proir_prob = self.calc_prior_prob()
        cond_prob = self.calc_cond_prob()
        post_prob = []
        for x in test_X:
            p = 0
            for c in proir_prob:
                prob = proir_prob[c]
                d = 0
                for a in x:
                    # print(c,cond_prob[c][d])
                    prob *= cond_prob[c][d][str(a)]
                    d += 1
                if prob > p:
                    p = prob
                    y = c
            post_prob.append((y, p))
        return post_prob

    # 定义预测函数
    def predict(self, test_X):
        post_prob = self.calc_post_prob(test_X)
        return agrmax(post_prob)


if __name__ == '__main__':
    X = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    ]).T
    Y = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])
    nbc = NaiveBayesClassifier(X, Y)
    # 输出先验概率
    # print(nbc.calc_prior_prob())
    # 打印条件概率
    # print(nbc.calc_cond_prob())
    # 测试
    test_X = np.array(([2, 'S'], [2, 'M']))
    print(nbc.calc_post_prob(test_X))
    print(nbc.predict(test_X))

# 待改进之处
# calc_freq函数可以优化，传入参数，这样在计算后验概率时可以调用，避免多层循环嵌套，提高代码的可阅读性
# 由于采用字典保存概率，当输入的特征向量中的某一个分量在原数据中不存在时，则无法进行计算
