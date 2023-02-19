from math import sqrt

import numpy as np
from numpy.random import default_rng
from numpy.linalg import pinv, norm
from scipy.stats import multivariate_normal

from .matrix_gamp import matrix_GAMP


def gk_expect_mixed_linear(Z_k_Ybar, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Compute E[Z|Zk, Ybar] for the mixed linear regression model. """
    E_Z_given_Zk_Ybar_cbar = np.zeros((L, L))
    P_Zk_Ybar_given_cbar = np.zeros((L))
    for i in range(L):
        # Recomputing S_ba and S_a every time the function is not optimal but left for simplicity now
        S_ba = np.hstack((S_12, S_11[i, :][:, None]))
        # Covariance of the distribution Zk, Ybar given cbar
        S_a = np.block([
            [S_22, S_21[:, i][:, None]],
            [S_12[i, :][:, None].T, S_11[i, i] + sigma_sq],
        ])
        E_Z_given_Zk_Ybar_cbar[:, i] = S_ba @ pinv(S_a) @ Z_k_Ybar
        try:
            P_Zk_Ybar_given_cbar[i] = multivariate_normal.pdf(
                Z_k_Ybar, mean=np.zeros(L+1), cov=S_a, allow_singular=True)
        except ValueError:
            print("Terminating, cov matrix not psd:\n", S_a)
            return np.full([L], np.nan)
    P_cbar_given_Zk_Ybar = alpha * P_Zk_Ybar_given_cbar
    if norm(P_cbar_given_Zk_Ybar, 1) == 0.0:
        # if P_cbar_given_Zk_Ybar is all zeros, then S_a is singular so return a uniform pdf
        P_cbar_given_Zk_Ybar = np.ones(L) / L
    else:
        P_cbar_given_Zk_Ybar = P_cbar_given_Zk_Ybar / norm(P_cbar_given_Zk_Ybar, 1)
    return E_Z_given_Zk_Ybar_cbar @ P_cbar_given_Zk_Ybar.T


def run_MLR_trial(p, L, n, alpha, B_row_cov, sigma_sq, n_iters, RNG=None):
    """
    Generate a random mixed linear regression dataset and then perform GAMP.
    Parameters:
        p: int = number of dimensions
        L: int = number of mixture components
        n: int = number of samples
        alpha: L x 1 = categorical distribution on components
        B_row_cov: L x L = covariance matrix of the rows of B
        sigma: int = noise level
        n_iters: int = max number of AMP iterations to perform
    Returns:
        B = true signal matrix
        B_hat_list = list of B_hat estimates for each AMP iteration
    """
    if RNG is None:
        RNG = default_rng()
    # initialise B signal matrix
    # rows of B are generated iid from joint Gaussian
    B = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    B_hat_0 = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    # initial estimate of B is generated from the same distribution

    # generate X iid Gaussian from N(0, 1/N)
    X = RNG.normal(0, np.sqrt(1 / n), (n, p))
    Theta = X @ B
    # generate class label latent variables from Cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, n), np.cumsum(alpha))

    # generate Y by picking elements from Theta according to c
    Y = np.take_along_axis(Theta, c[:, None], axis=1)
    Y = Y + RNG.normal(0, sqrt(sigma_sq), n)[:, None]

    B_hat_list, M_k_B_list = matrix_GAMP(X, Y, B_hat_0, B_row_cov, sigma_sq, alpha, n_iters, gk_expect_mixed_linear)
    return B, B_hat_list, M_k_B_list
