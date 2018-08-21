# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt


def test():
    xs = np.linspace(1e-6, 1, 300)
    ys = -xs * np.log(xs)
    plt.plot(xs, ys)
    plt.scatter(0.5, -0.5 * np.log(0.5), c='r')
    plt.show()

    
if __name__ == '__main__':
    test()