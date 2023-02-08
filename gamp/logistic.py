from math import sqrt

import numpy as np
from numpy.random import default_rng
from scipy.stats import norm as normal

from .gamp import GAMP

RNG = default_rng(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gk_expect_logistic(Z_k_and_Y, Sigma_k, sigma_sq):
    """ Compute E[Z|Zk, Ybar] for the logistic regression model. """
    mu_1 = Z_k_and_Y[0] * Sigma_k[0, 1] / Sigma_k[1, 1]
    sigma_1_sq = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    l = sqrt(np.pi / 8)
    P_Y_1_given_Z_k = normal.cdf(mu_1 / sqrt(l ** -2 + sigma_1_sq))
    const = l / sqrt(1 + l ** 2 * sigma_1_sq)
    inner = mu_1 * normal.cdf(const * mu_1) + const * sigma_1_sq * normal.pdf(const * mu_1)
    if Z_k_and_Y[1] == 1:
        return inner / P_Y_1_given_Z_k
    elif Z_k_and_Y[1] == 0:
        return (mu_1 - inner) / (1 - P_Y_1_given_Z_k)


def run_logistic_trial(p, n, sigma_beta_sq, n_iters):
    """
    Generate a logistic regression dataset and then perform GAMP.
    Parameters:
        p: int = number of dimensions
        n: int = number of samples
        sigma_beta_sq: float = signal variance
        n_iters: int = max number of AMP iterations to perform
    Returns:
        beta = true signal
        beta_hat_list = list of beta_hat estimates for each AMP iteration
        mu_k_list = list of state evolution mean for each AMP iteration
    """
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    beta = RNG.normal(0, sqrt(sigma_beta_sq), p)
    u = RNG.uniform(0, 1, n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_list, mu_k_list = GAMP(X, y, beta_hat_k, sigma_beta_sq, 0, n_iters, gk_expect_logistic)
    return beta, beta_hat_list, mu_k_list
