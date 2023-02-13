from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from numpy.linalg import inv
from scipy.stats import norm as normal

from .linear import gk_expect_linear
from .gamp import GAMP

RNG = default_rng(1)


def compute_P_Y_given_Z_k_omega(Y, Z_k, omega, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq):
    """ Compute p(Y | Z_k, omega) """
    if omega == 0:
        return normal.pdf(Y, 0, sqrt(sigma_sq))
    elif omega == 1:
        C_1 = np.exp(mu_1 ** 2 / (2 * sigma_1_sq)) * (1 - normal.cdf(0, mu_1, sqrt(sigma_1_sq)))
        exponent = - Y ** 2 / (2 * sigma_sq) + mu_2 ** 2 / (2 * sigma_2_sq)
        const = (1 - normal.cdf(0, mu_2, sqrt(sigma_2_sq))) / (C_1 * sqrt(2 * np.pi) * sqrt(sigma_sq + sigma_1_sq))
        return const * np.exp(exponent)


def compute_P_omega_given_Z_k(omega, Z_k, mu_1, sigma_1_sq):
    """ Compute p(omega | Z_k) """
    if omega == 0:
        return normal.cdf(0, mu_1, sqrt(sigma_1_sq))
    elif omega == 1:
        return 1 - normal.cdf(0, mu_1, sqrt(sigma_1_sq))


def compute_P_omega_given_Z_k_Y(omega, Z_k, Y, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq):

    P_omega_0_given_Z_k = compute_P_omega_given_Z_k(0, Z_k, mu_1, sigma_1_sq)
    P_omega_1_given_Z_k = 1 - P_omega_0_given_Z_k

    P_Y_given_Z_k_omega_1 = compute_P_Y_given_Z_k_omega(Y, Z_k, 1, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq)
    P_Y_given_Z_k_omega_0 = compute_P_Y_given_Z_k_omega(Y, Z_k, 0, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq)

    P_omega_0_given_Z_k_Y = P_omega_0_given_Z_k * P_Y_given_Z_k_omega_0 / \
        (P_omega_0_given_Z_k * P_Y_given_Z_k_omega_0 + P_omega_1_given_Z_k * P_Y_given_Z_k_omega_1)
    if omega == 0:
        return P_omega_0_given_Z_k_Y
    if omega == 1:
        return (1 - P_omega_0_given_Z_k_Y)


def gk_expect_relu(Z_k_and_Y, Sigma_k, sigma_sq):
    """ Compute E[Z|Zk, Ybar] for the rectified linear model. """
    Z_k, Y = Z_k_and_Y
    mu_1 = Z_k * Sigma_k[0, 1] / Sigma_k[1, 1]
    sigma_1_sq = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    sigma_1 = sqrt(sigma_1_sq)

    mu_2 = (Y * sigma_1_sq + mu_1 * sigma_sq) / (sigma_sq + sigma_1_sq)
    sigma_2_sq = sigma_sq * sigma_1_sq / (sigma_sq + sigma_1_sq)
    sigma_2 = sqrt(sigma_2_sq)

    P_omega_0_given_Z_k_Y = compute_P_omega_given_Z_k_Y(0, Z_k, Y, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq)
    P_omega_1_given_Z_k_Y = 1 - P_omega_0_given_Z_k_Y

    E_Z_given_Z_k_Y_omega_0 = mu_1 - sigma_1_sq * normal.pdf(0, mu_1, sigma_1) / normal.cdf(0, mu_1, sigma_1)
    E_Z_given_Z_k_Y_omega_1 = mu_2 + sigma_2_sq * normal.pdf(0, mu_2, sigma_2) / (1 - normal.cdf(0, mu_2, sigma_2))
    return P_omega_0_given_Z_k_Y * E_Z_given_Z_k_Y_omega_0 + P_omega_1_given_Z_k_Y * E_Z_given_Z_k_Y_omega_1


def run_relu_trial(p, n, sigma_sq, sigma_beta_sq, n_iters):
    """
    Generate a rectified linear regression dataset and then perform GAMP.
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
    y = np.clip(X @ beta, 0, np.inf) + eps
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_list, mu_k_list = GAMP(X, y, beta_hat_k, sigma_beta_sq, sigma_sq, n_iters, gk_expect_relu)
    return beta, beta_hat_list, mu_k_list


def run_relu_threshold_trial(p, n, sigma_sq, sigma_beta_sq, n_iters, beta=None):
    """
    Generate a rectified linear regression dataset and then perform GAMP.
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
    if beta is None:
        beta = RNG.normal(0, sqrt(sigma_beta_sq), p)
    y = np.clip(X @ beta, 0, np.inf) + eps
    # discard half the data (about half has been rectified)
    thresh = np.median(y)
    X = X[y > thresh, :]
    y = y[y > thresh]
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_list, mu_k_list = GAMP(X, y, beta_hat_k, sigma_beta_sq, sigma_sq, n_iters, gk_expect_linear)
    return beta, beta_hat_list, mu_k_list
