import numpy as np
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class GMM:
    def __init__(self, X, K, method=1):
        self.X = X
        self.K = K
        (self.n, self.d) = X.shape
        self.pi = np.full(shape=self.K, fill_value=1 / self.K)
        random_row = np.random.randint(low=0, high=self.n, size=self.K)
        self.mu = np.array([X[row_index, :] for row_index in random_row])
        self.method = method
        if method == 1:
            self.S = np.array([np.random.random(self.d) for _ in range(self.K)])
        elif method == 2:
            self.S = np.array([np.random.random(1) for _ in range(self.K)])
        self.r = np.zeros((self.n, self.K))
        self.maxiter = 500
        self.l = np.zeros(self.maxiter)
        self.tol = 1e-8
        self.stopiter = 0

    def fit(self):
        for iter in range(self.maxiter):
            self.stopiter = iter
            # for i in range(self.n):
            #     for k in range(self.K):
            #         self.r[i][k] = (self.pi[k] / (np.sqrt(np.linalg.det(self.S[k])))) * np.exp(
            #             -0.5 * np.dot(np.dot((self.X[i] - self.mu[k]).T, np.linalg.inv(self.S[k])),
            #                           (self.X[i] - self.mu[k])))
            #     self.l[iter] -= np.log(self.r[i].sum())
            #     self.r[i] /= self.r[i].sum()
            for k in range(self.K):
                det = 0
                exp = np.zeros(self.d)
                if self.method == 1:
                    # add a small constant to avoid numerical issue
                    det = np.prod(self.S[k]) + 1e-8
                    exp = (self.X - self.mu[k]) * (self.X - self.mu[k]) * (1 / self.S[k])
                    # add a small constant to avoid numerical issue
                    exp = np.exp(-0.5 * np.sum(exp, axis=1)) + 1e-8
                # self.r[:, k] = (self.pi[k] / (np.sqrt(np.linalg.det(self.S[k])))) * np.diag(np.exp(
                #     -0.5 * np.dot(np.dot((self.X - self.mu[k]), np.linalg.inv(self.S[k])),
                #                   (self.X - self.mu[k]).T)))
                elif self.method == 2:
                    det = self.S[k] ** self.d + 1e-8
                    exp = (self.X - self.mu[k]) * (self.X - self.mu[k]) * (1 / self.S[k])
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
                # mu_k = np.zeros(self.d)
                # s_k = np.zeros((self.d, self.d))
                # for i in range(self.n):
                #     mu_k += self.r[i][k] * self.X[i] / r_k
                #     s_k += self.r[i][k] * np.diag(self.X[i] * self.X[i].T) / r_k
                # self.mu[k] = mu_k
                # self.S[k] = s_k - np.diag(self.mu[k] * self.mu[k].T)
                self.mu[k] = np.sum(self.X * self.r[:, k].reshape((-1, 1)), axis=0) / r_k
                # self.S[k] = np.diag(np.sum(self.X * self.X * self.r[:, k].reshape((-1, 1)), axis=0) / r_k) - \
                #             np.diag(self.mu[k] * self.mu[k].T)
                if self.method == 1:
                    self.S[k] = np.sum(self.X * self.X * self.r[:, k].reshape((-1, 1)), axis=0) / r_k - \
                                self.mu[k] * self.mu[k].T
                elif self.method == 2:
                    self.S[k] = np.mean(np.sum(self.X * self.X * self.r[:, k].reshape((-1, 1)), axis=0) / r_k -
                                        self.mu[k] * self.mu[k].T)

        return self.l[:self.stopiter + 1]

    def predict(self):
        return np.argmax(self.r, axis=1)

    def cal_density(self, X):
        r = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            det = 0
            exp = np.zeros(self.d)
            if self.method == 1:
                det = np.prod(self.S[k]) + 1e-8
                exp = (X - self.mu[k]) * (X - self.mu[k]) * (1 / self.S[k])
                exp = np.exp(-0.5 * np.sum(exp, axis=1))
            elif self.method == 2:
                det = self.S[k] ** self.d
                exp = (X - self.mu[k]) * (X - self.mu[k]) * (1 / self.S[k])
                exp = np.exp(-0.5 * np.sum(exp, axis=1))
            r[:, k] = (self.pi[k] / np.sqrt(det)) * exp
        r_i = np.sum(r, axis=1)
        return r_i


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.mixture import GaussianMixture
    import time

    # iris = load_iris()
    # X = iris.data
    # y = iris.target
    # gmm = GMM(X, 3, method=2)
    # loss = gmm.fit()
    #
    # plt.figure()
    # plt.plot(loss)
    # plt.show()
    # pred = gmm.predict()
    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    # plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], c=pred)
    # plt.show()
    test_accuracy = []
    # K = [i for i in range(1, 11)]
    K = [5]
    loss_5 = []
    for k_component in K:
        train_set = datasets.MNIST(root='./data/', train=True, download=True)
        gmms = []
        sklearn_gmms = []
        losses = []
        Pr_y = []
        for i in range(10):
            print('Training model %i' % i)
            idx = train_set.targets == i
            X = train_set.data[idx].numpy().reshape((-1, 28 * 28)) / 255
            X += np.random.normal(0, 1, X.shape)
            Pr_y.append(len(train_set.data[idx].numpy()) / len(train_set.data))
            gmms.append(GMM(X, k_component))
            losses.append(gmms[-1].fit())

            sklearn_gmm = GaussianMixture(n_components=5, covariance_type='diag', tol=1e-8, max_iter=500)
            sklearn_gmm.fit(X)
            sklearn_gmms.append(sklearn_gmm)

            print('my log-likelihood: %f, sklearn log-likelihood: %f' % (losses[-1][-1], -sklearn_gmm.score_samples(X).sum()))
        pass
        if k_component == 5:
            loss_5 = losses
        test_set = datasets.MNIST(root='./data/', train=False, download=True)
        testX = test_set.data.numpy().reshape((-1, 28 * 28)) / 255
        testX += np.random.normal(0, 1, testX.shape)
        testy = test_set.targets.numpy()
        test_result = []
        sklearn_test_result = []
        for i in range(10):
            test_result.append(gmms[i].cal_density(testX) * Pr_y[i])
            proba = sklearn_gmms[i].predict_proba(testX)
            sklearn_test_result.append(np.sum(proba, axis=1) * Pr_y[i])
            pass
        test_pred = np.argmax(test_result, axis=0)
        sklearn_pred = np.argmax(sklearn_test_result, axis=0)
        correct = 0
        sklearn_correct = 0
        n = len(testy)
        for i in range(n):
            if testy[i] == test_pred[i]:
                correct += 1
            if testy[i] == sklearn_pred[i]:
                sklearn_correct += 1

        print('K = %i, Test accuracy: %f' % (k_component, (correct / n)))
        print('K = %i, Test accuracy: %f' % (k_component, (sklearn_correct / n)))
        test_accuracy.append((correct / n))

    plt.figure()
    plt.xlabel('K')
    plt.ylabel('Test accuracy')
    plt.plot(K, test_accuracy)
    plt.figure()
    for i in range(10):
        plt.plot(loss_5[i])
    plt.show()
    pass
