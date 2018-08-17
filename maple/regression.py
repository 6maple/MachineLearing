# coding = utf-8
import numpy as np
import loss


class LinearRegression:
    """线性回归
    
    """
    def __init__(self, xs, ys, optimizer, loss_func=loss.bin_square_error):
        """初始化
        
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        :param weights:                   回归参数
        :param optimizer:                 优化器
        :param loss_func:                 损失函数
        """
        self.xs = xs
        self.ys = ys
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.weights = np.random.random_sample(self.xs.shape[1]+1).astype(np.float32)[:, np.newaxis]
    
    @staticmethod
    def compute_weights(xs, ys):
        """根据公式计算最优参数
        
        weights = (X.T·X).I·X.T·y
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return np.dot( np.mat(np.dot(_xs.transpose(), _xs)).I, np.dot(_xs.transpose(), ys))
        
    @staticmethod
    def __loss_func(weights, xs, ys, loss_func):
        """用于优化器的损失函数
        
        :param weights:                   回归参数
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        :param loss_func:                 损失函数
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return loss_func(ys, _xs.dot(weights))
        
    def regress_once(self):
        """进行一次回归
        
        """
        self.optimizer.minimize(self.__loss_func, self.weights, [self.xs, self.ys, self.loss_func])
        self.weights = self.optimizer.optimize_once()
        return self.weights.copy()
    
    def predict(self, xs):
        """预测
        
        根据当前回归得出的参数进行预测
        :param xs:                        样本的输入
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return _xs.dot(self.weights)
    
    def compute_loss(self):
        """计算误差
        
        :return : 当前的误差
        """
        return self.loss_func(self.ys, self.predict(self.xs))
        
        
class LocallyWeightedLinearRegression:
    """局部加权线性回归
    
    """
    def __init__(self, xs, ys, tau=0.1):
        """初始化
        
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        :param tau:                       局部加权的波长参数
        """
        self.xs = xs
        self.ys = ys
        self.tau = tau
        self.weights = None
        
    @staticmethod
    def compute_locally_weight(x, xs, tau=0.1):
        n = xs.shape[0]
        W = np.eye(n, dtype=np.float32)
        denominator = 2 * (tau**2)
        for i in range(n):
            mat = np.mat(xs[i] - x)
            distance = (mat.T * mat)
            W[i, i] = np.exp(-distance[0, 0] / denominator)
        
        return W
    
    @staticmethod
    def compute_weights(xs, ys, W):
        """根据公式计算最优参数
        
        weights = (X.T·W·X).I·X.T·W·y
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return np.dot( np.mat(np.dot(np.dot(_xs.transpose(), W), _xs)).I, np.dot(np.dot(_xs.transpose(), W), ys))
    
    def __predict_one(self, x):
        """预测一个点
        
        :param x:                         目标点的输入
        """
        W = LocallyWeightedLinearRegression.compute_locally_weight(x, self.xs, self.tau)
        weights = LocallyWeightedLinearRegression.compute_weights(self.xs, self.ys, W)
        y = np.dot(np.mat(weights).T, x)
        return y[0, 0]
    
    def predict(self, xs):
        """预测
        
        根据当前回归得出的参数进行预测
        :param xs:                        样本的输入
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        for x in _xs:
            yield self.__predict_one(x)
    
    def compute_loss(self):
        """计算误差
        
        :return : 当前的误差
        """
        return self.loss_func(self.ys, self.predict(self.xs))