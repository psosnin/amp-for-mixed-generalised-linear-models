from math import sqrt

import numpy as np
from scipy.stats import norm as normal

"""
implement the Bayes-optimal g_k* for the logistic regression model
"""


def apply_gk(theta, y, Sigma_k, sigma_sq):
    """ Apply Bayesian-Optimal g_k* to each entry in theta and y. """

    # declare constants
    sigma_1_sq = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    l = sqrt(np.pi / 8)
    gamma = l / sqrt(1 + l ** 2 * sigma_1_sq)
    E_Z_given_Z_k = mu_1 = theta * Sigma_k[0, 1] / Sigma_k[1, 1]
    Var_Z_given_Z_k = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]

    # compute E[Z|Zk, Ybar] for each entry in theta and y
    P_Y_1_given_Z_k = normal.cdf(gamma * mu_1)
    numerator = mu_1 * normal.cdf(gamma * mu_1) + gamma * sigma_1_sq * normal.pdf(gamma * mu_1)
    E_Z_given_Z_k_Y_1 = numerator / P_Y_1_given_Z_k  # expectation of Z given Z_k and Y = 1
    E_Z_given_Z_k_Y_0 = (mu_1 - numerator) / (1 - P_Y_1_given_Z_k)  # expectation of Z given Z_k and Y = 0
    E_Z_given_Z_k_Y = np.where(y == 1, E_Z_given_Z_k_Y_1, E_Z_given_Z_k_Y_0)  # expectation of Z given Z_k and Y

    return (E_Z_given_Z_k_Y - E_Z_given_Z_k) / Var_Z_given_Z_k
