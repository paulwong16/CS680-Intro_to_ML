import numpy as np
import matplotlib.pyplot as plt


def compute_gradient(Xi, yi, w, b, C):
    tmp = yi * (np.dot(w.T, Xi) + b)
    if 1 - tmp[0] > 0:
        gw = -2 * C * yi * (1 - yi * (np.dot(w.T, Xi) + b)) * Xi
        gb = -1 * C * yi * (1 - yi * (np.dot(w.T, Xi) + b))
    else:
        gw, gb = np.zeros((2, 1)), 0
    gw = gw.reshape((2, 1))
    gb = gb[0]
    return gw, gb


def sgd_svm(X, y, w, b, max_pass, eta, C):
    n = X.shape[0]
    d = X.shape[1]
    for t in range(max_pass):
        for i in range(n):
            tmp = y[i] * (np.dot(X[i], w) + b)
            if tmp[0] <= 1:
                gw, gb = compute_gradient(X[i], y[i], w, b, C)
                w = w - eta * gw
                b = b - eta * gb
            w = (1. / (1. + eta)) * w
        loss = (1. / 2) * np.dot(w.T, w) + C * sum(
            [(max((1 - y[i] * (np.dot(X[i], w) + b)), 0)) ** 2 for i in range(n)])
    return w, b


if __name__ == "__main__":
    X = np.array([[2, 1], [1, 2], [3, 1], [3, 2]])
    y = np.array([[1], [1], [-1], [-1]])

    w = np.zeros((2, 1))
    b = 0
    w, b = sgd_svm(X, y, w, b, 1000, 0.001, 100)
    pt1 = [(-b - 1 * w[1][0]) / w[0][0], 1]
    pt2 = [(-b - 2 * w[1][0]) / w[0][0], 2]
    print('W is [%f, %f]^T, b is %f' % (w[0][0], w[1][0], b))
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
    plt.show()
