import numpy as np
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        (self.n, self.d) = X.shape
        self.pi = np.full(shape=self.K, fill_value=1 / self.K)
        random_row = np.random.randint(low=0, high=self.n, size=self.K)
        self.mu = np.array([X[row_index, :] for row_index in random_row])
        self.S = np.array([np.random.random(1) for _ in range(self.K)])
        self.r = np.zeros((self.n, self.K))
        self.maxiter = 500
        self.l = np.zeros(self.maxiter)
        self.tol = 1e-5
        self.stopiter = 0

    def fit(self):
        for iter in range(self.maxiter):
            self.stopiter = iter
            for k in range(self.K):
                # add a small constant to avoid numerical issue
                det = self.S[k] ** self.d + 1e-8
                exp = (self.X - self.mu[k]) * (self.X - self.mu[k]) * (1 / self.S[k])
                # add a small constant to avoid numerical issue
                exp = np.exp(-0.5 * np.sum(exp, axis=1))
                self.r[:, k] = (self.pi[k] / np.sqrt(det)) * exp
            r_i = np.sum(self.r, axis=1)
            self.l[iter] = - np.sum(np.log(r_i))
            r_i = r_i.reshape((-1, 1))
            self.r /= r_i
            if iter > 0 and np.abs(self.l[iter] - self.l[iter - 1]) <= self.tol * np.abs(self.l[iter]):
                break
            for k in range(self.K):
                r_k = self.r[:, k].sum()
                self.pi[k] = r_k / self.n
                self.mu[k] = np.sum(self.X * self.r[:, k].reshape((-1, 1)), axis=0) / r_k
                self.S[k] = np.mean(np.sum(self.X * self.X * self.r[:, k].reshape((-1, 1)), axis=0) / r_k -
                                    self.mu[k] * self.mu[k].T)
        return self.l[:self.stopiter + 1]

    def predict(self):
        return np.argmax(self.r, axis=1)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target
    gmm = GMM(X, 3)
    loss = gmm.fit()

    plt.figure()
    plt.plot(loss)
    plt.ylabel('negative log-likelihood')
    plt.xlabel('iter')
    plt.show()
    pred = gmm.predict()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=pred)
    plt.show()
    pass
