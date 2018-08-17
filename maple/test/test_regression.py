# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from regression import LinearRegression, LocallyWeightedLinearRegression
from optimize import GradientDescentOptimizer

np.random.seed(1)

def test_linear_regression():
    # define sample data
    xs = np.linspace(0, 2, 50).astype(np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.5, xs.shape).astype(np.float32)
    ys = xs + noise
    # define figure
    fig = plt.figure('linear regression')
    # show the sample data
    ax = fig.add_subplot(1,1,1,title='linear regression')
    ax.scatter(xs, ys, c='r', label='sample points')
    # show real curve
    _ys = xs
    ax.plot(xs, _ys, 'g', label='real curve')
    # define linear regression operation
    sgd_optimizer = GradientDescentOptimizer(0.008, 1)
    linear_regression = LinearRegression(xs, ys, sgd_optimizer)
    # start regression
    ax.plot(xs, linear_regression.predict(xs), 'b', label='regression curve')
    plt.legend(loc='upper right')
    plt.ion()
    for step in range(50):
        plt.pause(0.01)
        linear_regression.regress_once()
        ax.lines.pop()
        ax.plot(xs, linear_regression.predict(xs), 'b', label='regression curve')
        print('step {}: loss = {}'.format(step, linear_regression.compute_loss()))
    
    print('\n', '-'*20, 'optimize end', '-'*20, '\n')
    
    print('weights : \n{}'.format(linear_regression.weights))
    print('loss : \n{}'.format(linear_regression.compute_loss()))
    print('-'*50, '\n')
    
    print('compute directly: ')
    linear_regression.weights = LinearRegression.compute_weights(xs, ys)
    print('weights: \n{}'.format(linear_regression.weights))
    print('loss: \n{}'.format(linear_regression.compute_loss()))
    print('-'*50, '\n')
    
    print('real value: ')
    linear_regression.weights = np.array([1, 0], dtype=np.float32)[:, np.newaxis]
    print('weights: \n{}'.format(np.array([1, 0], dtype=np.float32)[:, np.newaxis]))
    print('loss : \n{}'.format(linear_regression.compute_loss()))
    print('-'*50, '\n')
    
    ax.plot(xs, linear_regression.predict(xs), 'g', label='real curve')
    
    plt.ioff()
    plt.savefig('linear regression.png')
    plt.show()


def test_locally_weighted_linear_regression():
    # define sample data
    xs = np.linspace(0, 2, 50).astype(np.float32)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, xs.shape).astype(np.float32)
    ys = ((xs**3)/3 - xs**2 + (3/4) * xs) + noise
    # define figure
    fig = plt.figure('locally weighted linear regression')
    # show the sample data
    ax = fig.add_subplot(1,1,1,title='locally weighted linear regression')
    ax.scatter(xs, ys, c='r', label='sample points')
    ax.plot(xs, ((xs**3)/3 - xs**2 + (3/4) * xs), 'g', label='real curve')
    
    # define locally weighted linear regression operation
    locally_weighted_linear_regression = LocallyWeightedLinearRegression(xs, ys, 0.2)
    pys = locally_weighted_linear_regression.predict(xs)
    pys = list(pys)
    # show regression curve
    ax.plot(xs, pys, 'b', label='regression curve : tau=0.2')
    
    plt.ioff()
    plt.legend(loc='upper right')
    plt.savefig('locally weighted linear regression.png')
    plt.show()

    
if __name__ == '__main__':
    test_linear_regression()
    #test_locally_weighted_linear_regression()