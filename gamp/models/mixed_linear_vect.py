from math import sqrt

import numpy as np
from numpy.random import default_rng
from numpy.linalg import pinv, norm, inv
from scipy.stats import multivariate_normal

from ..fitting.matrix_gamp import matrix_GAMP


def apply_gk_mixed_linear(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Apply Bayesian-Optimal g_k* to each row of Theta_k """
    cov_Z_given_Zk = S_11 - S_12 @ pinv(S_22) @ S_21
    Theta_k_Ybar = np.hstack((Theta_k, Y))

    cov_Zk_Ybar_given_cbar = np.zeros((L, L+1, L+1))
    cov_ZK_Ybar_and_Z_given_cbar = np.zeros((L, L, L+1))
    for i in range(L):
        cov_ZK_Ybar_and_Z_given_cbar[i, :, :] = np.hstack(
            (S_12, S_11[i, :][:, None]))
        cov_Zk_Ybar_given_cbar[i, :, :] = np.block([
            [S_22, S_21[:, i][:, None]],
            [S_12[i, :][:, None].T, S_11[i, i] + sigma_sq],
        ])

    E_Z_given_Zk = (S_12 @ pinv(S_22) @ Theta_k_Ybar[:, : L].T).T
    E_Z_given_Zk_Ybar_cbar = cov_ZK_Ybar_and_Z_given_cbar @ pinv(
        cov_Zk_Ybar_given_cbar) @ Theta_k_Ybar.T

    try:
        P_Zk_Ybar_given_cbar = np.array([multivariate_normal.pdf(Theta_k_Ybar, mean=np.zeros(L+1), cov=cov,
                                                                 allow_singular=True) for cov in cov_Zk_Ybar_given_cbar])
    except ValueError:
        print("Warning, cov matrix is not PSD")
        return np.nan * np.zeros_like(Theta_k)

    P_cbar_given_Zk_Ybar = alpha[:, None] * P_Zk_Ybar_given_cbar
    # handle singular S_a (all zero pdf)
    P_cbar_given_Zk_Ybar[:, np.all(P_cbar_given_Zk_Ybar == 0, axis=0)] = 1
    P_cbar_given_Zk_Ybar = P_cbar_given_Zk_Ybar / \
        norm(P_cbar_given_Zk_Ybar, 1, axis=0)
    E_Z_given_Zk_Ybar = (E_Z_given_Zk_Ybar_cbar.T @
                         P_cbar_given_Zk_Ybar[None].T)[:, :, 0]

    return (E_Z_given_Zk_Ybar - E_Z_given_Zk) @ inv(cov_Z_given_Zk)


def generate_data_mixed_linear(p, L, n, alpha, B_row_cov, sigma_sq, RNG):
    """
    Generate a random mixed linear regression dataset.
    Parameters:
        p: int = number of dimensions
        L: int = number of mixture components
        n: int = number of samples
        alpha: L x 1 = categorical distribution on components
        B_row_cov: L x L = covariance matrix of the rows of B
        RNG: numpy random number generate
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        B: p x L = signal matrix
    """
    if RNG is None:
        RNG = default_rng()
    # initialise B signal matrix
    # rows of B are generated iid from joint Gaussian
    B = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    # initial estimate of B is generated from the same distribution

    # generate X iid Gaussian from N(0, 1/N)
    X = RNG.normal(0, np.sqrt(1 / n), (n, p))
    Theta = X @ B
    # generate class label latent variables from Cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, n), np.cumsum(alpha))

    # generate Y by picking elements from Theta according to c
    Y = np.take_along_axis(Theta, c[:, None], axis=1)
    Y = Y + RNG.normal(0, sqrt(sigma_sq), n)[:, None]

    return X, Y, B


def run_MLR_trial(p, L, n, alpha, B_row_cov, sigma_sq, n_iters, RNG=None, return_data=False):
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

    B_hat_list, M_k_B_list = matrix_GAMP(
        X, Y, B_hat_0, B_row_cov, sigma_sq, alpha, n_iters, apply_gk_mixed_linear)
    if return_data:
        return B, B_hat_list, M_k_B_list, X, Y, B_hat_0
    return B, B_hat_list, M_k_B_list
