# coding = utf-8
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('../')

from optimize import GradientDescentOptimizer


def func(_x, _y):
    return ( (1-_y**5+_x**5)*np.exp(-_x**2-_y**2) ).astype(np.float32)


def optimize_func(weights):
    return func(*weights)


def test_SGD_minimize():
    ### data area < ###
    X = np.linspace(-1,3,20)
    Y = np.linspace(-1,3,20)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    ### data area > ###
    ### plt figure < ###
    fig = plt.figure('test_SGD_minimize')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.gist_earth)
    # set view port
    ax.view_init(elev=66., azim=56.)
    ### plt figure > ###
    ### gradient descent > ###
    # define weights
    weights = np.array([1.481, 0.4], dtype=np.float32)
    # define optimizer
    sgd_optimizer = GradientDescentOptimizer(0.1)
    sgd_optimizer.minimize(optimize_func, weights, [])
    # start gradient descent
    # ax.scatter(weights[0], weights[1], optimize_func(weights)+0.1, c='r')
    plt.ion()
    for step in range(50):
        plt.pause(0.1)
        tmp = weights.copy()
        weights = sgd_optimizer.optimize_once()
        #  print(optimize_func(weights))
        ax.plot([tmp[0], weights[0]], [tmp[1], weights[1]], [optimize_func(tmp), optimize_func(weights)], c='r')
        print('step {}:{}'.format(step+1, optimize_func(weights)))
    
    ### gradient descent > ###
    print('\n', '-'*20, 'optimize end', '-'*20)
    plt.ioff()
    plt.savefig('test_case1.png')
    plt.show()


def test_SGD_maximize():
    ### data area < ###
    X = np.linspace(-1,3,20)
    Y = np.linspace(-1,3,20)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    ### data area > ###
    ### plt figure < ###
    fig = plt.figure('test_SGD_maximize')
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.gist_earth)
    # set view port
    ax.view_init(elev=66., azim=56.)
    ### plt figure > ###
    ### gradient descent > ###
    # define weights
    weights = np.array([0.014119354397058581, 1.6495296075940131], dtype=np.float32)
    # define optimizer
    sgd_optimizer = GradientDescentOptimizer(0.1)
    sgd_optimizer.maximize(optimize_func, weights, [])
    # start gradient descent
    # ax.scatter(weights[0], weights[1], optimize_func(weights)+0.1, c='r')
    plt.ion()
    for step in range(50):
        plt.pause(0.1)
        tmp = weights.copy()
        weights = sgd_optimizer.optimize_once()
        #  print(optimize_func(weights))
        ax.plot([tmp[0], weights[0]], [tmp[1], weights[1]], [optimize_func(tmp), optimize_func(weights)], c='r')
        print('step {}:{}'.format(step+1, optimize_func(weights)))
    
    ### gradient descent > ###
    print('\n', '-'*20, 'optimize end', '-'*20)
    plt.ioff()
    plt.savefig('test_case2.png')
    plt.show()
    
    
if __name__ == '__main__':
    test_SGD_minimize()
    test_SGD_maximize()
    
    