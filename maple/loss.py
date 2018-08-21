# coding = utf-8
import numpy as np


def mean_square_error(ys, pys):
    """均方误差
    
    :param ys:                   实际值
    :param pys:                  预测值
    """
    return np.mean(np.square(ys - pys))
    
    
def bin_square_error(ys, pys):
    """二乘方误差
    
    :param ys:                   实际值
    :param pys:                  预测值
    """
    return (ys - pys).transpose().dot(ys - pys) / 2

    
def cross_entryopy(ys, pys):
    """交叉熵
    
    :param ys:                   实际值
    :param pys:                  预测值
    """
    result = 0
    for i in range(len(ys)):
        result += ys[i].transpose().dot(np.log(pys[i]))
    return -result