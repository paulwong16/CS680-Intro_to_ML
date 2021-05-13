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


def l2_distance(x1,x2):
    return np.linalg.norm((x1-x2),ord=2,axis=1)


def l1_distance(x1,x2):
    return np.linalg.norm((x1-x2),ord=1,axis=1)


def aggregate(neighbors):
    dict = {}
    n = len(neighbors)
    for i in range(n):
        if neighbors[i] in dict:
            dict[neighbors[i]] += 1
        else:
            dict[neighbors[i]] = 1
    max_count = 0
    res = float('-inf')
    for i in dict.keys():
        if (max_count < dict[i]):
            res = i
            max_count  = dict[i]
    return res


def knn(trainX,trainY,testX,k,dist,testY):
    n = trainX.shape[0]
    testYhat = np.zeros((testX.shape[0],1))
    acc = 0
    for i in range(testX.shape[0]):
        if dist == 2:
            d = l2_distance(trainX, testX[i])
        elif dist == 1:
            d = l1_distance(trainX, testX[i])
        ind = np.unravel_index(np.argsort(d), d.shape)
        neighbors = list(np.reshape(trainY[ind[0][:k]], k))
        testYhat[i] = aggregate(neighbors)
        if testYhat[i] == testY[i]:
            acc += 1
    error = (testX.shape[0] - acc) / testX.shape[0]
    return (testYhat, error)


if __name__ == '__main__':
    np.random.seed(0)
    for sigma in [0.01, 0.1, 1, 10]:
        n = 6000
        d = 2
        w = np.zeros((d, 1))
        w[0] = 1
        b = 0
        [X, y] = data_gen(n, d, sigma, w, b)
        X_train = X[:3000]
        y_train = y[:3000]
        X_test = X[3000:]
        y_test = y[3000:]
        bayes_error = Bayes(X_test, y_test, w, b)
        [yhat, onenn_error] = knn(X_train, y_train, X_test, 1, 2, y_test)
        print('When sigma = %f, Bayes error: %f, 1-NN error: %f' % (sigma, bayes_error, onenn_error))
