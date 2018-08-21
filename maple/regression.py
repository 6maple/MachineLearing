# coding = utf-8
import numpy as np
import loss
from optimize import GradientDescentOptimizer


class LinearRegression:
    """线性回归
    
    """
    def __init__(self, xs, ys, optimizer, loss_func=loss.bin_square_error):
        """初始化
        
        :param xs:                        样本的输入
        :param ys:                        样本的输出
        :param weights:                   回归参数
        :param optimizer:                 优化器
        :param loss_func:                 损失函数
        """
        self.xs = xs
        self.ys = ys
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.weights = np.random.random((self.xs.shape[1]+1,1)).astype(np.float32)
        self.optimizer.minimize(self.__loss_func, self.weights, [self.xs, self.ys, self.loss_func])
    
    @staticmethod
    def compute_weights(xs, ys):
        """根据公式计算最优参数（这里采用的是二乘方损失函数）
        
        weights = (X.T·X).I·X.T·y
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        :return : 最优参数
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
        :return : 损失值
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return loss_func(ys, _xs.dot(weights))
        
    def regress_once(self):
        """进行一次回归
        
        :return : 优化后的参数
        """
        self.weights = self.optimizer.optimize_once()
        return self.weights
    
    def predict(self, xs):
        """预测
        
        根据当前回归得出的参数进行预测
        :param xs:                        样本的输入
        :return : 预测值
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return _xs.dot(self.weights)
    
    def compute_loss(self):
        """计算误差
        
        :return : 损失值
        """
        return self.loss_func(self.ys, self.predict(self.xs))
        
        
class LocallyWeightedLinearRegression:

    """局部加权线性回归
    
    """
    def __init__(self, xs, ys, tau=0.1):
        """初始化
        
        :param xs:                        样本的输入
        :param ys:                        样本的输出
        :param tau:                       局部加权的波长参数
        """
        self.xs = xs
        self.ys = ys
        self.tau = tau
        self.weights = None
        
    @staticmethod
    def compute_locally_weight(x, xs, tau=0.1):
        """计算局部权值
        
        :param x:                         目标点的输入
        :param xs:                        样本的输入
        :param tau:                       局部加权的波长参数
        :return : 局部权值
        """
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
        :return : 最优参数
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return np.dot( np.mat(np.dot(np.dot(_xs.transpose(), W), _xs)).I, np.dot(np.dot(_xs.transpose(), W), ys))
    
    def __predict_one(self, x):
        """预测一个点
        
        :param x:                         目标点的输入
        :return : 预测值
        """
        W = LocallyWeightedLinearRegression.compute_locally_weight(x, self.xs, self.tau)
        weights = LocallyWeightedLinearRegression.compute_weights(self.xs, self.ys, W)
        y = np.dot(np.mat(weights).T, x)
        return y[0, 0]
    
    def predict(self, xs):
        """预测
        
        根据当前回归得出的参数进行预测
        :param xs:                        样本的输入
        :return : 预测值
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        for x in _xs:
            yield self.__predict_one(x)
    
    def compute_loss(self):
        """计算误差
        
        :return : 损失值
        """
        return self.loss_func(self.ys, self.predict(self.xs))
        
        
class LogisticRegression:
    """逻辑回归
    
    主要用于二分类问题，即输出为{0,1}的情况
    """
    def __init__(self, xs, ys, learning_rate=0.1, batch_size=-1):
        """初始化
        
        :param xs:                        样本的输入
        :param ys:                        样本的输出
        :param batch_size:                 每次优化使用的样本数量
        :param learning_rate:             学习效率
        """
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.weights = np.random.random((self.xs.shape[1]+1,1)).astype(np.float32)
        self.optimizer = GradientDescentOptimizer(learning_rate, batch_size, self.gradient_func)
        self.optimizer.maximize(None, self.weights, [self.xs, self.ys])
        
    @staticmethod
    def logistic_func(x):
        """logistic函数
        
        :param x:                         输入
        :return : x对应的值
        """
        # for perceptron algorithm
        #result = x.copy()
        #for i in range(x.shape[0]):
        #    for j in range(x.shape[1]):
        #        if x[i, j] < 10:
        #            result[i, j] = 0
        #        else:
        #            result[i, j] = 1.
        #            
        #return result
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def gradient_func(func, weights, i, inputs, offset=1e-3):
        """估计目标函数的梯度
        
        :param func:                       目标函数
        :param weights:                    参数
        :param i:                          梯度方向
        :param inputs:                     目标函数的输入
        :param offset:                     估计梯度所用到的点邻域宽度
        :return: 目标函数在weights[i]方向上的梯度
        """
        _xs = np.insert(inputs[0], inputs[0].shape[1], 1, axis=1)
        pys = LogisticRegression.logistic_func(_xs.dot(weights))
        gradient = (inputs[1] - pys).transpose().dot(_xs[:, i])
        # 使用梯度上升来极大化似然函数
        return -gradient
        
    
    def regress_once(self):
        """进行一次回归
        
        :return : weights
        """
        self.weights = self.optimizer.optimize_once()
        return self.weights
    
    def predict(self, xs):
        """预测
        
        根据当前回归得出的参数进行预测
        :param xs:                        样本的输入
        :return : predict ys
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        return self.logistic_func(_xs.dot(self.weights))
        
    def compute_loss(self):
        """计算误差
        
        :return : loss
        """
        # for perceptron algorithm
        #return loss.mean_square_error(self.ys, self.predict(self.xs))
        
        _ys = np.insert(self.ys, 1, 0, axis=1)
        pys = np.insert(self.predict(self.xs), 1, 0, axis=1)
        for i in range(len(_ys)):
            _ys[i, 1] = 1 - _ys[i, 0]
            pys[i, 1] = 1 - pys[i, 0]
            
        return loss.cross_entryopy(_ys, pys)
        
       
class SoftmaxRegression:
    """softmax回归
    
    主要用于二分类问题，即输出为{0,1}的情况
    """
    def __init__(self, xs, ys, learning_rate=0.1, batch_size=-1):
        """初始化
        
        :param xs:                        样本的输入
        :param ys:                        样本的输出
        :param batch_size:                 每次优化使用的样本数量
        :param learning_rate:             学习效率
        """
        self.xs = xs
        self.ys = ys
        self.learning_rate = learning_rate
        self.weights = np.random.random((self.ys.shape[1],self.xs.shape[1]+1)).astype(np.float32)
        self.optimizer = GradientDescentOptimizer(learning_rate, batch_size, self.gradient_func)
        self.optimizer.maximize(None, self.weights, [self.xs, self.ys])
        
    @staticmethod
    def softmax_func(weights, x, k=-1):
        """softmax函数
        
        :param weights:                   参数 shape(n, m)
        :param x:                         目标输入 shape(m,)
        :return : x对应的值（用于预测） (n,)
        """
        n = weights.shape[0]
        # 分母
        denominator = np.sum(np.exp(weights.dot(x.transpose())))
        # 只输出某一类的值
        if 0 <= k < n:
            numerator = np.exp(weights[k].dot(x.transpose()))
            return numerator / denominator
        # 输出所有可能的类别的值
        py = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            # 分子
            numerator = np.exp(weights[i].dot(x.transpose()))
            py[i] = numerator / denominator
        
        return py

    @staticmethod
    def gradient_func(func, weights, i, inputs, offset=1e-3):
        """估计目标函数的梯度
        
        :param func:                       目标函数
        :param weights:                    参数
        :param i:                          梯度方向
        :param inputs:                     目标函数的输入
        :param offset:                     估计梯度所用到的点邻域宽度
        :return: 目标函数在weights[i]方向上的梯度
        """
        #delta = 0.01 #正则化项
        
        gradient = 0
        _xs = np.insert(inputs[0], inputs[0].shape[1], 1, axis=1)
        _ys = inputs[1]
        n = _xs.shape[0]
        for j in range(n):
            if _ys[j][i] == 1:
                gradient += _xs[j] * (1 - SoftmaxRegression.softmax_func(weights, _xs[j], i)) #+ delta * weights[i]
            else:
                gradient += _xs[j] * ( - SoftmaxRegression.softmax_func(weights, _xs[j], i)) #+ delta * weights[i]
            
        # 使用梯度上升来极大化似然函数
        return -gradient
        
    
    def regress_once(self):
        """进行一次回归
        
        :return : weights
        """
        self.weights = self.optimizer.optimize_once()
        return self.weights
    
    def predict(self, xs):
        """预测
        
        根据当前回归得出的参数进行预测
        :param xs:                        样本的输入
        :return : predict ys
        """
        _xs = np.insert(xs, xs.shape[1], 1, axis=1)
        pys = []
        for _x in _xs:
            tmp = self.softmax_func(self.weights, _x)
            pys.append(tmp)
            
        return np.array(pys, dtype=np.float32)
        
    def compute_loss(self):
        """计算误差
        
        :return : loss
        """
            
        return loss.cross_entryopy(self.ys, self.predict(self.xs))