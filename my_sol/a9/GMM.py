import numpy as np
import matplotlib.pyplot as plt
from scipy import special


class GMM:
    def __init__(self, pi, mu, sigma):
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

    def gen_normal_1(self):
        return np.random.normal(self.mu[0], self.sigma[0], 1)

    def gen_normal_2(self):
        return np.random.normal(self.mu[1], self.sigma[1], 1)


def GMMsample(gmm, n=1000, b=50):
    X = np.zeros(n)
    for i in range(n):
        X[i] = gmm.gen_normal_1() if np.random.random(1) < gmm.pi else gmm.gen_normal_2()
    plt.figure()
    n, bins, patches = plt.hist(X, b)
    plt.show()
    return X


def Phi(X, mu, sigma):
    return 0.5 * (1 + special.erf(((X - mu) / (sigma * np.sqrt(2)))))


def F_(X, gmm):
    return gmm.pi * Phi(X, gmm.mu[0], gmm.sigma[0]) + (1 - gmm.pi) * Phi(X, gmm.mu[1], gmm.sigma[1])


def norm_Phi_inv(Y):
    return special.erfinv(2 * Y - 1) * np.sqrt(2)


def GMMinv(X, gmm, b=50):
    F_X = gmm.pi * Phi(X, gmm.mu[0], gmm.sigma[0]) + (1 - gmm.pi) * Phi(X, gmm.mu[1], gmm.sigma[1])
    U = norm_Phi_inv(F_X)
    plt.figure()
    n, bins, patches = plt.hist(U, b)
    plt.show()
    return U


def BinarySearch(F, u, lb=-100, ub=100, maxiter=100, tol=1e-8):
    gmm = GMM(0.5, [1, -1], [0.5, 0.5])
    x = None
    while F(lb, gmm) > u:
        ub = lb
        lb = lb / 2
    while F(ub, gmm) < u:
        lb = ub
        ub = ub * 2
    for i in range(maxiter):
        x = (lb + ub) / 2
        t = F(x, gmm)
        if t > u:
            ub = x
        else:
            lb = x
        if np.abs(t - u) <= tol:
            break
    return x


def plot_T(Z=None):
    if Z is None:
        Z = np.linspace(-5, (-5 + 0.1 * 100), 100, endpoint=False)
    T = np.zeros(Z.shape)
    for i in range(len(Z)):
        T[i] = BinarySearch(F_, Phi(Z[i], 0, 1))
    plt.figure()
    plt.plot(Z, T)
    plt.show()


def PushForward(Z, gmm):
    X = np.zeros(Z.shape)
    for i in range(len(Z)):
        X[i] = BinarySearch(F_, Phi(Z[i], 0, 1))
    plt.figure()
    n, bins, patches = plt.hist(X, 50)
    plt.show()
    return X


if __name__ == '__main__':
    gmm = GMM(0.5, [1, -1], [0.5, 0.5])
    X = GMMsample(gmm, 1000, 50)
    U = GMMinv(X, gmm, 50)
    plot_T()
    Z = np.zeros(1000)
    for i in range(1000):
        Z[i] = np.random.normal(0, 1)
    X_ = PushForward(Z, gmm)
    U_ = GMMinv(X_, gmm, 50)
    pass
