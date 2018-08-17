# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from regression import LinearRegression
from optimize import GradientDescentOptimizer

np.random.seed(1)

def test_linear_regression():
    # define sample data
    xs = np.linspace(0, 2, 20).astype(np.float32)[:, np.newaxis]
    bias = np.random.normal(0, 0.5, xs.shape).astype(np.float32)
    ys = xs + bias
    # define figure
    fig = plt.figure('test_linear_regression')
    # show the sample data
    ax = fig.add_subplot(1,1,1)
    ax.scatter(xs, ys, c='r')
    # define linear regression operation
    sgd_optimizer = GradientDescentOptimizer(0.02, 1)
    linear_regression = LinearRegression(xs, ys, sgd_optimizer)
    # start regression
    plt.ion()
    ax.plot(xs, linear_regression.predict(xs), 'b')
    for step in range(50):
        plt.pause(0.01)
        linear_regression.regress_once()
        ax.lines.pop()
        ax.plot(xs, linear_regression.predict(xs), 'b')
        print('step {}: loss = {}'.format(step, linear_regression.compute_loss()))
    
    print('\n', '-'*20, 'optimize end', '-'*20)
    print('weights : \n{}'.format(linear_regression.weights))
    linear_regression.weights = np.array([1, 0], dtype=np.float32)[:, np.newaxis]
    print('real value\'s loss : \n{}'.format(linear_regression.compute_loss()))
    plt.ioff()
    plt.show()
    

if __name__ == '__main__':
    test_linear_regression()