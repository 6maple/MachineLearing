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