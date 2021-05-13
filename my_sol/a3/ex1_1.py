import numpy as np

def data_gen(n,d,sigma,w,b):
    X = np.random.rand(n,d)*2-1
    U = np.random.rand(n,1)
    sgm = 1./(1+np.exp(-((np.dot(X,w)+b)/sigma)))
    y = np.zeros((n,1))
    for i in range(n):
        y[i] = 1 if U[i] <= sgm[i] else -1
    return [X,y]


if __name__ == '__main__':
    np.random.seed(0)
    n = 6000
    d = 2
    sigma = 1
    w = np.zeros((d, 1))
    w[0] = 1
    b = 0
    [X, y] = data_gen(n, d, sigma, w, b)
