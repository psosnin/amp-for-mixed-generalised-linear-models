import numpy as np
from numpy.linalg import inv

"""
implement the Bayes-optimal g_k* for the linear regression model
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
    """ Compute E[Z|Zk, Ybar] for the linear regression model. """
    S_a = np.array([Sigma_k[0, 1], Sigma_k[0, 0]])
    S_ba = np.array([
        [Sigma_k[1, 1], Sigma_k[0, 1]],
        [Sigma_k[0, 1], Sigma_k[0, 0] + sigma_sq]
    ])
    E_Z_given_Z_k_Y = S_a @ inv(S_ba) @ Z_k_and_Y
    return E_Z_given_Z_k_Y
