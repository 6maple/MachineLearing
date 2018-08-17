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
    def __loss_func(weights, xs, ys, loss_func):
        """用于优化器的损失函数
        
        :param weights:                   回归参数
        :param xs:                        样本的输入
        :param xs:                        样本的输出
        :param loss_func:                 损失函数
        """
        xs = np.insert(xs.copy(), xs.shape[1], 1, axis=1)
        return loss_func(ys, xs.dot(weights))
        
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
        xs = np.insert(xs.copy(), xs.shape[1], 1, axis=1)
        return xs.dot(self.weights)
    
    def compute_loss(self):
        """计算误差
        
        :return : 当前的误差
        """
        return self.loss_func(self.ys, self.predict(self.xs))