# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from regression import LinearRegression, LocallyWeightedLinearRegression, LogisticRegression, SoftmaxRegression
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
    plt.legend(loc='upper left')
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

    
def test_logistic_regression():
    N = 70
    TN = 30
    c1 = (1, 1)
    c2 = (2, 2)
    R = 0.5
    xs1 = np.random.normal(c1[0], R, (N,))
    ys1 = np.random.normal(c1[1], R, (N,))
    xs2 = np.random.normal(c2[0], R, (N,))
    ys2 = np.random.normal(c2[1], R, (N,))
    # define figure
    fig = plt.figure('logistic regression')
    ax = fig.add_subplot(1, 1, 1, title='logistic regression')
    # show sample
    ax.scatter(xs1, ys1, c='b', label='class 1')
    ax.scatter(xs2, ys2, c='r', label='class 2')
    plt.legend(loc='upper left')
    # define inputs
    xs = np.array(list(zip(xs1, ys1)) + list(zip(xs2, ys2)), dtype=np.float32)
    ys = np.array([0]*N + [1]*N, dtype=np.float32)[:, np.newaxis]
    X, Y = np.meshgrid(np.linspace(0,3,TN), np.linspace(0,3,TN))
    Z = np.zeros(X.shape, dtype=np.float32)
    # define logistic regression
    logistic_regression = LogisticRegression(xs, ys, 0.01)
    # start regress
    plt.ion()
    for step in range(50):
        plt.pause(0.01)
        logistic_regression.regress_once()
        
        print('step {}: loss = {}'.format(step, logistic_regression.compute_loss()))
        
        # show
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                vxs = np.array([X[i, j], Y[i, j]], dtype=np.float32)[np.newaxis, :]
                Z[i, j] = logistic_regression.predict(vxs)
                #tmp = logistic_regression.predict(vxs)
                #if tmp < 0.5:
                #    tmp = 0
                #else:
                #    tmp = 1
                #Z[i, j] = tmp
        ax.collections = ax.collections[:2]
        ax.contour(X, Y, Z, 1, colors='g')
        
    print('\n', '-'*20, 'optimize end', '-'*20, '\n')
    #print('predict : \n{}'.format(logistic_regression.predict(xs).transpose()))
    print('loss : {}'.format(logistic_regression.compute_loss()), '\n')
    # block
    plt.ioff()
    plt.savefig('logistic regression.png')
    plt.show()

   
def test_softmax_regression():
    N = 70
    TN = 30
    c1 = (1, 1)
    c2 = (2, 2)
    c3 = (1, 3)
    R = 0.5
    xs1 = np.random.normal(c1[0], R, (N,))
    ys1 = np.random.normal(c1[1], R, (N,))
    xs2 = np.random.normal(c2[0], R, (N,))
    ys2 = np.random.normal(c2[1], R, (N,))
    xs3 = np.random.normal(c3[0], R, (N,))
    ys3 = np.random.normal(c3[1], R, (N,))
    # define figure
    fig = plt.figure('logistic regression')
    ax = fig.add_subplot(1, 1, 1, title='logistic regression')
    # show sample
    ax.scatter(xs1, ys1, c='b', label='class 1')
    ax.scatter(xs2, ys2, c='r', label='class 2')
    ax.scatter(xs3, ys3, c='g', label='class 3')
    plt.legend(loc='upper right')
    # define inputs
    xs = np.array(list(zip(xs1, ys1)) + list(zip(xs2, ys2)) + list(zip(xs3, ys3)), dtype=np.float32)
    ys = np.array([[1, 0, 0]]*N+[[0, 1, 0]]*N+[[0, 0, 1]]*N, dtype=np.float32)#[:, np.newaxis]
    #xs = np.array(list(zip(xs1, ys1)) + list(zip(xs2, ys2)), dtype=np.float32)
    #ys = np.array([[1, 0]]*N+[[0, 1]]*N, dtype=np.float32)#[:, np.newaxis]
    X, Y = np.meshgrid(np.linspace(0,3,TN), np.linspace(0,3,TN))
    Z = np.zeros(X.shape, dtype=np.float32)
    # define logistic regression
    softmax_regression = SoftmaxRegression(xs, ys, 0.001)
    # start regress
    plt.ion()
    for step in range(200):
        plt.pause(0.01)
        softmax_regression.regress_once()
        
        print('step {}: loss = {}'.format(step, softmax_regression.compute_loss()))
        
        # show
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                vxs = np.array([X[i, j], Y[i, j]], dtype=np.float32)[np.newaxis, :]
                Z[i, j] = np.argmax(softmax_regression.predict(vxs))
                
        ax.collections = ax.collections[:3]
        #ax.collections = ax.collections[:2]
        ax.contour(X, Y, Z, 2, colors='black')
        
    print('\n', '-'*20, 'optimize end', '-'*20, '\n')
    print('predict : \n{}'.format(softmax_regression.predict(xs)))
    print('loss : {}'.format(softmax_regression.compute_loss()), '\n')
    # block
    plt.ioff()
    plt.savefig('softmax regression.png')
    plt.show()

    
if __name__ == '__main__':
    #test_linear_regression()
    #test_locally_weighted_linear_regression()
    #test_logistic_regression()
    test_softmax_regression()