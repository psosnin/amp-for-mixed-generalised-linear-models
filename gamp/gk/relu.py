from math import sqrt

import numpy as np
from scipy.stats import norm as normal


"""
implement the Bayes-optimal g_k* for the rectified linear regression model
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
