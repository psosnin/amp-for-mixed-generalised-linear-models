from math import sqrt

import numpy as np
from numpy.random import default_rng
from numpy.linalg import pinv, norm
from scipy.stats import multivariate_normal
from scipy.stats import norm as normal

from .gamp import GAMP

RNG = default_rng(1)


def gk_expect_logistic(Z_k_Ybar, S_11, S_12, S_21, S_22, L, sigma, alpha):
    """ Compute E[Z|Zk, Ybar] for the mixed logistic regression model. """
    # E_Z_given_Zk_Ybar = sum(P_cbar_given_Zk_Ybar * E_Z_given_Zk_Ybar_cbar)
    assert L == 1
    Z_k = Z_k_Ybar[0]
    Ybar = Z_k_Ybar[1]
    mu_1 = Z_k * S_12[0, 0] / S_22[0, 0]
    sigma_1 = sqrt(S_11[0] - S_12[0] ** 2 / S_22[0])
    l = sqrt(np.pi / 8)
    # print(f"Ybar={Ybar}, Zk={Z_k}, mu_1={mu_1}, sig1={sigma_1}")
    zeta = sigma_1 * l / sqrt(1 + mu_1 ** 2 * l ** 2)
    # print(f"zeta={zeta}")
    P_Ybar1_given_Zk = normal.cdf(mu_1 / sqrt(l ** -2 + sigma_1 ** 2))
    # print(f"P(Ybar=1 | Zk)={P_Ybar1_given_Zk}")
    tmp = zeta * normal.pdf(zeta) + normal.cdf(zeta)
    if Ybar == 1:
        return mu_1 * tmp / P_Ybar1_given_Zk
    elif Ybar == 0:
        return mu_1 * (1 - tmp) / (1 - P_Ybar1_given_Zk)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def run_logistic_trial(p, L, n, alpha, sigma, n_iters):
    """
    Generate a random mixed logistic regression dataset and then perform GAMP.
    Parameters:
        p: int = number of dimensions
        L: int = number of mixture components
        n: int = number of samples
        alpha: L x 1 = categorical distribution on components
        sigma: int = noise level
        n_iters: int = max number of AMP iterations to perform
    Returns:
        B = true signal matrix
        B_hat_list = list of B_hat estimates for each AMP iteration
    """

    # initialise B signal matrix
    # rows of B are generated iid from joint Gaussian
    B_row_mean = np.zeros(L)
    B_row_cov = np.eye(L)
    B = RNG.multivariate_normal(B_row_mean, B_row_cov, p)
    B_hat_0 = RNG.multivariate_normal(B_row_mean, B_row_cov, p)
    # initial estimate of B is generated from the same distribution

    # generate X iid Gaussian from N(0, 1/N)
    X = RNG.normal(0, np.sqrt(1 / n), (n, p))
    Theta = X @ B
    # generate class label latent variables from Cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, n), np.cumsum(alpha))

    # probability of Y depends on sigmoid
    P_Y_1 = sigmoid(np.take_along_axis(Theta, c[:, None], axis=1))
    u = RNG.uniform(0, 1, n)[:, None]
    Y = np.array(u < P_Y_1, dtype=int)
    B_hat_list, M_k_B_list = GAMP(X, Y, B_hat_0, B_row_cov, sigma, alpha, n_iters, gk_expect_logistic)
    return B, B_hat_list, M_k_B_list
