from math import sqrt

import numpy as np
from scipy.stats import norm as normal


"""
implement the Bayes-optimal g_k* for the rectified linear regression model
"""


def apply_gk(theta, y, Sigma_k, sigma_sq):
    """ Apply Bayesian-Optimal g_k* to each entry in theta and y. """

    Var_Z_given_Zk = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    E_Z_given_Zk = theta * Sigma_k[0, 1] / Sigma_k[1, 1]
    E_Z_given_Zk_Y = compute_E_Z_given_Zk_Y(theta, y, Sigma_k, sigma_sq)
    return (E_Z_given_Zk_Y - E_Z_given_Zk) / Var_Z_given_Zk


def compute_E_Z_given_Zk_Y(theta, y, Sigma_k, sigma_sq):
    """ Compute E[Z | Zk, Y]"""
    # compute constants
    sigma_1_sq = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    sigma_2_sq = sigma_sq * sigma_1_sq / (sigma_sq + sigma_1_sq)
    sigma, sigma_1, sigma_2 = sqrt(sigma_sq), sqrt(sigma_1_sq), sqrt(sigma_2_sq)
    mu_1 = theta * Sigma_k[0, 1] / Sigma_k[1, 1]
    mu_2 = (y * sigma_1_sq + mu_1 * sigma_sq) / (sigma_sq + sigma_1_sq)

    # compute intermediate probabilities
    P_omega_0_given_Zk = normal.cdf(0, mu_1, sigma_1)
    P_omega_1_given_Zk = 1 - P_omega_0_given_Zk
    P_Y_given_Zk_omega_0 = normal.pdf(y, 0, sigma)
    P_Y_given_Zk_omega_1 = compute_P_Y_given_Zk_omega_1(y, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq)
    P_omega_0_given_Zk_Y = (
        P_omega_0_given_Zk * P_Y_given_Zk_omega_0 /
        (P_omega_0_given_Zk * P_Y_given_Zk_omega_0 + P_omega_1_given_Zk * P_Y_given_Zk_omega_1)
    )
    P_omega_1_given_Zk_Y = 1 - P_omega_0_given_Zk_Y

    E_Z_given_Zk_Y_omega_0 = mu_1 - sigma_1_sq * normal.pdf(0, mu_1, sigma_1) / normal.cdf(0, mu_1, sigma_1)
    E_Z_given_Zk_Y_omega_1 = mu_2 + sigma_2_sq * normal.pdf(0, mu_2, sigma_2) / (1 - normal.cdf(0, mu_2, sigma_2))

    return P_omega_0_given_Zk_Y * E_Z_given_Zk_Y_omega_0 + P_omega_1_given_Zk_Y * E_Z_given_Zk_Y_omega_1


def compute_P_Y_given_Zk_omega_1(Y, sigma_sq, mu_1, sigma_1_sq, mu_2, sigma_2_sq):
    """ Compute p(Y | Zk, omega = 1) """
    C_1 = np.exp(mu_1 ** 2 / (2 * sigma_1_sq)) * (1 - normal.cdf(0, mu_1, sqrt(sigma_1_sq)))
    exponent = - Y ** 2 / (2 * sigma_sq) + mu_2 ** 2 / (2 * sigma_2_sq)
    const = (1 - normal.cdf(0, mu_2, sqrt(sigma_2_sq))) / (C_1 * sqrt(2 * np.pi) * sqrt(sigma_sq + sigma_1_sq))
    return const * np.exp(exponent)
