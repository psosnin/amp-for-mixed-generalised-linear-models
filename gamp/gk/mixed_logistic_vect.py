from math import sqrt

import numpy as np
from numpy.linalg import pinv, norm, inv
from scipy.stats import norm as normal


"""
implement a vectorised version of the Bayes-optimal g_k* for the mixed logistic regression model.
Runs much faster than the original version but is less readable.
"""


def apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Apply Bayesian-Optimal g_k* to each row of Theta_k """
    n, L = Theta_k.shape
    l = sqrt(np.pi / 8)

    cov_Z_given_Zk = S_11 - S_12 @ pinv(S_22) @ S_21
    Theta_k_Y = np.hstack((Theta_k, Y))
    E_Z_given_Zk = (S_12 @ pinv(S_22) @ Theta_k_Y[:, : L].T).T

    P_Y_1_given_Zk_c = normal.cdf(E_Z_given_Zk / np.sqrt(l ** -2 + np.diag(cov_Z_given_Zk)))
    P_Y_given_Zk_c = Y * P_Y_1_given_Zk_c + (1 - Y) * (1 - P_Y_1_given_Zk_c)
    P_c_given_Zk_Y = alpha * P_Y_given_Zk_c
    P_c_given_Zk_Y = P_c_given_Zk_Y / norm(P_c_given_Zk_Y, 1, axis=1)[:, None]

    cov_Zj_Zi_Zk = compute_cov_Zj_Zi_Zk(S_11, S_12, S_21, S_22)
    cov_Zj_given_Zi_Zk = compute_cov_Zj_given_Zi_Zk(cov_Zj_Zi_Zk)
    alpha_ji, gamma_ji = compute_alpha_and_gamma_ji(cov_Zj_Zi_Zk, cov_Zj_given_Zi_Zk, Theta_k)
    eta_jik = compute_eta_jik(alpha_ji, gamma_ji, E_Z_given_Zk)

    E_Z_given_Zk_Y = compute_E_Z_given_Zk_Y(
        Y, alpha, cov_Z_given_Zk, E_Z_given_Zk, P_c_given_Zk_Y, P_Y_given_Zk_c, alpha_ji, eta_jik)

    return (E_Z_given_Zk_Y - E_Z_given_Zk) @ inv(cov_Z_given_Zk)


def compute_E_Z_given_Zk_Y(Y, alpha, cov_Z_given_Zk, E_Z_given_Zk, P_c_given_Zk_Y, P_Y_given_Zk_c, alpha_ji, eta_jik):
    """
    Compute E[Z|Zk, Ybar] for the mixed logistic regression model.
    """
    n = Y.size
    L = len(alpha)
    E_Z_given_Zk_Y = np.zeros((n, L))
    # argument of pdf and cdf in integral evaluation
    argument = eta_jik / np.sqrt(1 + alpha_ji ** 2 * np.diag(cov_Z_given_Zk))[None]
    pdf_eval = normal.pdf(argument)
    cdf_eval = normal.cdf(argument)

    pdf_const = (alpha_ji * np.diag(cov_Z_given_Zk) / np.sqrt(1 + alpha_ji ** 2 * np.diag(cov_Z_given_Zk)))
    integral = pdf_const * pdf_eval + E_Z_given_Zk[:, None, :] * cdf_eval

    E_Z_given_Y_Zk_c = (Y[:, :, None] * integral + (1 - Y[:, :, None]) *
                        (E_Z_given_Zk[:, None, :] - integral)) / P_Y_given_Zk_c[:, :, None]

    E_Z_given_Zk_Y = np.sum(E_Z_given_Y_Zk_c * P_c_given_Zk_Y[:, :, None], axis=1)

    return E_Z_given_Zk_Y


def compute_cov_Zj_Zi_Zk(S_11, S_12, S_21, S_22):
    """
    Compute the covariance matrix of p(Z_j, Z_i, Z^k) for each combination of j, i
    Returns an L x L x (L + 2) x (L + 2 matrix), where the first two indices index j and i
    """
    L = S_11.shape[0]

    cov_Zj_Zi_Zk = np.zeros((L, L, L+2, L+2))
    for j in range(L):
        for i in range(L):
            cov_Zj_Zi_Zk[j, i, :, :] = np.block([
                [S_11[j, j], S_11[j, i], S_12[j, :]],
                [S_11[i, j], S_11[i, i], S_12[i, :]],
                [S_21[:, j, None], S_21[:, i, None], S_22]
            ])
    return cov_Zj_Zi_Zk


def compute_cov_Zj_given_Zi_Zk(cov_Zj_Zi_Zk):
    """
    Compute the variance of p(Zj|Zi, Zk).
    Return an L x L matrix indexed by j, i respectively.
    """
    L = cov_Zj_Zi_Zk.shape[0]
    cov_Zj_given_Zi_Zk = np.zeros((L, L))
    for j in range(L):
        cov_Zj_given_Zi_Zk[j, :] = (
            cov_Zj_Zi_Zk[j, :, 0, 0] -
            (cov_Zj_Zi_Zk[j, :, 0, None, 1:] @
             (pinv(cov_Zj_Zi_Zk[j, :, 1:, 1:]) @ cov_Zj_Zi_Zk[j, :, 1:, 0, None]))[:, 0, 0]
        )
    cov_Zj_given_Zi_Zk = np.clip(cov_Zj_given_Zi_Zk, 0, np.inf)
    return cov_Zj_given_Zi_Zk


def compute_eta_jik(alpha_ji, gamma_ji, E_Z_given_Zk):
    """
    Compute eta_jik = alpha_ji * E_Z_i_given_Zk + gamma_ji
    Parameters:
        alpha_ji: L x L matrix
        gamma_ji: n x L x L matrix
        E_Z_given_Zk: n x L matrix
    Returns:
        eta: n x L x L: matrix indexed by n, j, i
    """
    n, L, _ = gamma_ji.shape
    eta = np.zeros((n, L, L))
    for i in range(L):
        for j in range(L):
            eta[:, j, i] = alpha_ji[j, i] * E_Z_given_Zk[:, i] + gamma_ji[:, j, i]
    return eta


def compute_alpha_and_gamma_ji(cov_Zj_Zi_Zk, cov_Zj_given_Zi_Zk, Zk):
    """
    Compute alpha_ji and gamma_ji where
        alpha_j|i * Zi + gamma_factor_j|i * Zk = mu_j|ik / sqrt(l ** -2 + sigma_j|ik_sq)
    Parameters:
        cov_Zj_Zi_Zk: L x L x (L + 2) x (L + 2) matrix
        cov_Zj_given_Zi_Zk: L x L matrix
        Zk: n x L matrix, rows are individual Zk
    Returns:
        alpha = L x L matrix indexed by j, i
        gamma_factor = n x L x L matrix indexed by n, j, i
    """
    n, L = Zk.shape
    alpha = np.zeros((L, L))
    l = sqrt(np.pi / 8)
    gamma = np.zeros((n, L, L))
    for j in range(L):
        mu_j_ik_factor = (pinv(cov_Zj_Zi_Zk[j, :, 1:, 1:]) @ cov_Zj_Zi_Zk[j, :, 0, 1:, None])[:, :, 0]
        alpha[j, :] = mu_j_ik_factor[:, 0] / np.sqrt(l ** -2 + cov_Zj_given_Zi_Zk[j, :])
        gamma[:, j, :] = ((mu_j_ik_factor[:, 1:] @ Zk[:, :, None]) /
                          np.sqrt(l ** -2 + cov_Zj_given_Zi_Zk[j, :])[:, None])[:, :, 0]
    return alpha, gamma
