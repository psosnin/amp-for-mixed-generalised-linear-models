import numpy as np
from numpy.linalg import inv

"""
implement the Bayes-optimal g_k* for the linear regression model
"""


def apply_gk(theta, y, Sigma_k, sigma_sq):
    """ Apply Bayesian-Optimal g_k* to each entry in theta and y. """
    S_a = np.array([Sigma_k[0, 1], Sigma_k[0, 0]])
    S_ba = np.array([
        [Sigma_k[1, 1], Sigma_k[0, 1]],
        [Sigma_k[0, 1], Sigma_k[0, 0] + sigma_sq]
    ])
    theta_and_y = np.vstack((theta, y))
    Var_Z_given_Z_k = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    E_Z_given_Z_k = theta * Sigma_k[0, 1] / Sigma_k[1, 1]
    E_Z_given_Z_k_Y = S_a @ inv(S_ba) @ theta_and_y
    gk = (E_Z_given_Z_k_Y - E_Z_given_Z_k) / Var_Z_given_Z_k
    return gk
