import numpy as np
from numpy import pi as pi
import pymc as pm
from pymc.gp.util import stabilize, cholesky, JITTER_DEFAULT

np.random.seed(7926)


def draw_routine_kernel(amp, ls, lkj_scale):
    # Draw corr:
    corrmat = 0.5 * np.eye(7)
    corrs = pm.LKJCorr.dist(n=7, eta=lkj_scale).eval()
    corrmat[np.triu_indices(7, k=1)] = corrs
    corrmat = corrmat + corrmat.T

    # Hour kernel:
    hours = np.array([[i] for i in range(1, 25)])
    hour_K = pm.gp.cov.ExpQuad(ls=ls, input_dim=1)(hours).eval()

    # Kronecker product of the two:
    K = amp**2 * np.kron(corrmat, hour_K)

    return K


def draw_alpha(n_cust, n_week, mean=-8, amp=1.5, ls=3):
    inputs = np.arange(1, n_week + 1)[:, np.newaxis]
    K = amp**2 * pm.gp.cov.Matern12(input_dim=1, ls=ls)(inputs)
    L = cholesky(stabilize(K))
    alpha = pm.MvNormal.dist(mu=mean, chol=L, shape=(n_cust, n_week)).eval()
    return alpha


def draw_gamma(n_cust, n_week, mean=-8, amp=1.5, ls=5):
    inputs = np.arange(1, n_week + 1)[:, np.newaxis]
    K = amp**2 * pm.gp.cov.Matern12(input_dim=1, ls=ls)(inputs)
    L = cholesky(stabilize(K))
    gamma = pm.MvNormal.dist(mu=mean, chol=L, shape=(n_cust, n_week)).eval()
    return gamma


def draw_mu(amp=2, ls=5, lkj_scale=2):
    K = draw_routine_kernel(amp=amp, ls=ls, lkj_scale=lkj_scale)
    L = cholesky(stabilize(K))
    mu = pm.MvNormal.dist(mu=0, chol=L).eval()
    return mu


def draw_eta(n_cust, amp=2, ls=5, lkj_scale=2):
    K = draw_routine_kernel(amp=amp, ls=ls, lkj_scale=lkj_scale)
    L = cholesky(stabilize(K))
    eta = pm.MvNormal.dist(mu=0, chol=L, shape=(n_cust, 168)).eval()
    return eta


class SimData:
    def __init__(self, n_cust, n_week):
        self.n_cust = n_cust
        self.n_week = n_week
        n_dayhour = 168

        self.alpha = draw_alpha(n_cust, n_week)
        self.gamma = draw_gamma(n_cust, n_week)
        self.mu = draw_mu()
        self.eta = draw_eta(n_cust)

        self.y = np.zeros((n_cust, n_week, n_dayhour))
        self.e_rand = np.zeros((n_cust, n_week, n_dayhour))
        self.e_rout = np.zeros((n_cust, n_week, n_dayhour))
        for i in range(n_cust):
            for w in range(n_week):
                for j in range(n_dayhour):
                    lambda_iwj = np.exp(self.alpha[i, w] + self.mu[j]) + np.exp(
                        self.gamma[i, w] + self.eta[i, j]
                    )
                    self.y[i, w, j] = np.random.poisson(lambda_iwj)
                    self.e_rand[i, w, j] = np.exp(self.alpha[i, w] + self.mu[j])
                    self.e_rout[i, w, j] = np.exp(self.gamma[i, w] + self.eta[i, j])


def draw_active_index(n_cust, n_week, churn_rate):
    active = np.ones((n_cust, n_week))
    for i in range(n_cust):
        for w in range(1, n_week):
            if active[i, w - 1] == 0:
                active[i, w] = 0
            else:
                active[i, w] = np.random.binomial(1, p=1 - churn_rate, size=1)
    return active


class SimDataChurn:
    def __init__(self, n_cust, n_week, churn_rate):
        self.n_cust = n_cust
        self.n_week = n_week
        n_dayhour = 168

        self.alpha = draw_alpha(n_cust, n_week)
        self.gamma = draw_gamma(n_cust, n_week)
        self.mu = draw_mu()
        self.eta = draw_eta(n_cust)
        self.active = draw_active_index(n_cust, n_week, churn_rate)

        self.y = np.zeros((n_cust, n_week, n_dayhour))
        for i in range(n_cust):
            for w in range(n_week):
                for j in range(n_dayhour):
                    if self.active[i, w] == 1:
                        lambda_iwj = np.exp(self.alpha[i, w] + self.mu[j]) + np.exp(
                            self.gamma[i, w] + self.eta[i, j]
                        )
                        self.y[i, w, j] = np.random.poisson(lambda_iwj)
