from math import sqrt

import numpy as np
from scipy.stats import norm as normal

"""
implement the Bayes-optimal g_k* for the logistic regression model
"""


def apply_gk(theta, y, Sigma_k, sigma_sq):
    """ Apply Bayesian-Optimal g_k* to each entry in theta and y. """
    Var_Z_given_Z_k = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    theta_and_y = np.vstack((theta, y))
    return np.apply_along_axis(compute_gk_1d, 0, theta_and_y, Var_Z_given_Z_k, Sigma_k, sigma_sq)


def compute_gk_1d(Z_k_and_Y, Var_Z_given_Z_k, Sigma_k, sigma_sq):
    """ Compute the optial gk given Z_k and Y. """
    E_Z_given_Z_k = Z_k_and_Y[0] * Sigma_k[0, 1] / Sigma_k[1, 1]
    E_Z_given_Z_k_Y = gk_expect(Z_k_and_Y, Sigma_k, sigma_sq)
    return (E_Z_given_Z_k_Y - E_Z_given_Z_k) / Var_Z_given_Z_k


def gk_expect(Z_k_and_Y, Sigma_k, sigma_sq):
    """ Compute E[Z|Zk, Ybar] for the logistic regression model. """
    mu_1 = Z_k_and_Y[0] * Sigma_k[0, 1] / Sigma_k[1, 1]
    sigma_1_sq = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    l = sqrt(np.pi / 8)
    P_Y_1_given_Z_k = normal.cdf(mu_1 / sqrt(l ** -2 + sigma_1_sq))
    const = l / sqrt(1 + l ** 2 * sigma_1_sq)
    inner = mu_1 * normal.cdf(const * mu_1) + const * \
        sigma_1_sq * normal.pdf(const * mu_1)
    if Z_k_and_Y[1] == 1:
        return inner / P_Y_1_given_Z_k
    elif Z_k_and_Y[1] == 0:
        return (mu_1 - inner) / (1 - P_Y_1_given_Z_k)
