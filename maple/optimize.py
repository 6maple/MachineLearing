# coding = utf-8
import numpy as np
import random


class GradientDescentOptimizer:
    """梯度下降优化器
    
    """
    def __init__(self, learning_rate, batch_size=-1):
        """初始化
        
        :param learning_rate:              学习率
        :param batch_size:                 每次优化使用的样本数量
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.func = None
        self.weights = None
        self.inputs = None
        
    def minimize(self, func, weights, inputs):
        """求最小化func的weights
        
        :param func:                       目标函数
        :param weights:                    需要调整的参数
        :param inputs:                     目标函数的输入
        :return: 调整好的参数
        """
        self.func = func
        self.weights = weights
        self.inputs = inputs
    
    def maximize(self, func, weights, inputs):
        """求最大化func的weights
        
        :param func:                       目标函数
        :param weights:                    需要调整的参数
        :param inputs:                     目标函数的输入
        :return: 调整好的参数
        """
        return self.minimize(lambda *args, **kwargs: -func(*args, **kwargs), weights, inputs)
    
    def optimize_once(self):
        """执行一次优化
        
        :return: 优化结果
        """
        weights = self.weights.copy()
        inputs = []
        # 按照batch_size选择样本
        if self.batch_size > 0 and inputs:
            # 随机选择batch_size个样本
            indices = list(range(len(self.inputs[0])))
            batch_indices = random.sample(indices, self.batch_size)
            for one_input in self.inputs:
                tmp = []
                for i in batch_indices:
                    tmp.append(one_input[i])
                inputs.append(tmp)
        else:
            # 使用所有样本
            inputs = self.inputs
        
        for i in range(len(weights)):
            self.weights[i] = weights[i] - self.learning_rate * self.estimate_gradient(self.func, weights, i, inputs)
        
        return self.weights.copy()
    
    @staticmethod
    def estimate_gradient(func, weights, i, inputs, offset=1e-3):
        """估计目标函数的梯度
        
        :param func:                       目标函数
        :param weights:                    参数
        :param i:                          梯度方向
        :param inputs:                     目标函数的输入
        :param offset:                     估计梯度所用到的点邻域宽度
        :return: 目标函数在weights[i]方向上的梯度
        """
        dweights = weights.copy()
        dweights[i] += offset
        delta = func(dweights, *inputs) - func(weights, *inputs)
        return delta / offset