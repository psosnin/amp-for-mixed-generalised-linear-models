from math import sqrt

import numpy as np
from numpy.random import default_rng
from numpy.linalg import inv

from ..fitting.gamp import GAMP

RNG = default_rng()


def gk_expect_linear(Z_k_and_Y, Sigma_k, sigma_sq):
    """ Compute E[Z|Zk, Ybar] for the linear regression model. """
    S_a = np.array([Sigma_k[0, 1], Sigma_k[0, 0]])
    S_ba = np.array([
        [Sigma_k[1, 1], Sigma_k[0, 1]],
        [Sigma_k[0, 1], Sigma_k[0, 0] + sigma_sq]
    ])
    E_Z_given_Z_k_Y = S_a @ inv(S_ba) @ Z_k_and_Y
    return E_Z_given_Z_k_Y


def run_linear_trial(p, n, sigma_sq, sigma_beta_sq, n_iters):
    """
    Generate a linear regression dataset and then perform GAMP.
    Parameters:
        p: int = number of dimensions
        n: int = number of samples
        sigma_sq: float = noise variance
        sigma_beta_sq: float = signal variance
        n_iters: int = max number of AMP iterations to perform
    Returns:
        beta = true signal
        beta_hat_list = list of beta_hat estimates for each AMP iteration
        mu_k_list = list of state evolution mean for each AMP iteration
    """
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    eps = RNG.normal(0, sqrt(sigma_sq), n)
    beta = RNG.normal(0, sqrt(sigma_beta_sq), p)
    y = X @ beta + eps
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_list, mu_k_list = GAMP(
        X, y, beta_hat_k, sigma_beta_sq, sigma_sq, n_iters, gk_expect_linear)
    return beta, beta_hat_list, mu_k_list
