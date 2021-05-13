import numpy as np

def data_gen(n,d,sigma,w,b):
    X = np.random.rand(n,d)*2-1
    U = np.random.rand(n,1)
    sgm = 1./(1+np.exp(-((np.dot(X,w)+b)/sigma)))
    y = np.zeros((n,1))
    for i in range(n):
        y[i] = 1 if U[i] <= sgm[i] else -1
    return [X,y]


def Bayes(X, y, w, b):
    n = X.shape[0]
    prediction = np.dot(X, w) + b
    y_pred = np.zeros((n, 1))
    error = 0
    for i in range(n):
        y_pred[i] = 1 if prediction[i] >= 0 else -1
        if y_pred[i] == y[i]:
            error += 1

    error = (n - error) / n
    return error


if __name__ == '__main__':
    np.random.seed(0)
    n = 6000
    d = 2
    sigma = 1
    w = np.zeros((d, 1))
    w[0] = 1
    b = 0
    for i in [10, 100, 1000, 10000, 100000, 1000000]:
        [X, y] = data_gen(i, d, sigma, w, b)
        print('When n = %i, bayes error: %f' % (i, Bayes(X, y, w, b)))
