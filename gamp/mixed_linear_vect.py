from math import sqrt

import numpy as np
from numpy.random import default_rng
from numpy.linalg import pinv, norm, inv
from scipy.stats import multivariate_normal

from .matrix_gamp import matrix_GAMP


def apply_gk_mixed_linear(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Apply Bayesian-Optimal g_k* to each row of Theta_k """
    cov_Z_given_Zk = S_11 - S_12 @ pinv(S_22) @ S_21
    Theta_k_Ybar = np.hstack((Theta_k, Y))
    S_12_S_22_inv = S_12 @ pinv(S_22)

    S_ba_S_a_inv_list = np.zeros((L, L, L+1))
    S_a_list = np.zeros((L, L+1, L+1))
    for i in range(L):
        S_ba = np.hstack((S_12, S_11[i, :][:, None]))
        S_a = np.block([
            [S_22, S_21[:, i][:, None]],
            [S_12[i, :][:, None].T, S_11[i, i] + sigma_sq],
        ])
        S_a_list[i, :, :] = S_a
        S_ba_S_a_inv_list[i, :, :] = S_ba @ pinv(S_a)

    E_Z_given_Zk = (S_12_S_22_inv @ Theta_k_Ybar[:, : L].T).T
    E_Z_given_Zk_Ybar_cbar = np.array(S_ba_S_a_inv_list) @ Theta_k_Ybar.T

    try:
        P_Zk_Ybar_given_cbar = np.array([multivariate_normal.pdf(Theta_k_Ybar, mean=np.zeros(L+1), cov=S_a,
                                                                 allow_singular=True) for S_a in S_a_list])
    except ValueError:
        print("Warning, cov matrix is not PSD")
        return np.nan * np.zeros_like(Theta_k)

    P_cbar_given_Zk_Ybar = alpha[:, None] * P_Zk_Ybar_given_cbar
    P_cbar_given_Zk_Ybar[:, np.all(P_cbar_given_Zk_Ybar == 0, axis=0)] = 1  # handle singular S_a
    P_cbar_given_Zk_Ybar = P_cbar_given_Zk_Ybar / norm(P_cbar_given_Zk_Ybar, 1, axis=0)
    E_Z_given_Zk_Ybar = (E_Z_given_Zk_Ybar_cbar.T @ P_cbar_given_Zk_Ybar[None].T)[:, :, 0]

    return (E_Z_given_Zk_Ybar - E_Z_given_Zk) @ inv(cov_Z_given_Zk)


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

    B_hat_list, M_k_B_list = matrix_GAMP(X, Y, B_hat_0, B_row_cov, sigma_sq, alpha, n_iters, apply_gk_mixed_linear)
    return B, B_hat_list, M_k_B_list
