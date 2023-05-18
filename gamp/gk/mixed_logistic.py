from math import sqrt

import numpy as np
from numpy.linalg import pinv, norm, inv
from scipy.stats import norm as normal

"""
implement the Bayes-optimal g_k* for the mixed logistic regression model
"""


def compute_gk_1d(Z_k_Ybar, S_11, S_12, S_21, S_22, cov_Z_given_Zk, L, sigma_sq, alpha):
    """ Compute the Bayesian-Optimal g_k* """
    E_Z_given_Zk = S_12 @ pinv(S_22) @ Z_k_Ybar[0:L]
    Sigma_k = np.block([[S_11, S_12], [S_21, S_22]])
    E_Z_given_Zk_Ybar = compute_E_Z_given_Zk_Y(Z_k_Ybar[:L], Z_k_Ybar[-1], Sigma_k, alpha)
    g_Zk_Ybar = inv(cov_Z_given_Zk) @ (E_Z_given_Zk_Ybar - E_Z_given_Zk)
    return g_Zk_Ybar


def apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Apply Bayesian-Optimal g_k* to each row of Theta_k """
    cov_Z_given_Zk = S_11 - S_12 @ pinv(S_22) @ S_21
    Theta_k_Ybar = np.hstack((Theta_k, Y))
    return np.apply_along_axis(compute_gk_1d, 1, Theta_k_Ybar, S_11, S_12, S_21, S_22, cov_Z_given_Zk, L, sigma_sq, alpha)


def compute_P_c_given_Zk_Y(Y, alpha, mu_zzk, Sigma_zzk):
    result = np.zeros_like(alpha)
    for i, a in enumerate(alpha):
        result[i] = a * compute_P_Y_given_Zk_c(Y, i, mu_zzk, Sigma_zzk)
    return result / norm(result, 1)


def compute_P_Y_given_Zk_c(Y, c, mu_zzk, Sigma_zzk):
    l = sqrt(np.pi / 8)
    mu_j_given_Zk = mu_zzk[c]
    sigma_j_given_Zk_sq = Sigma_zzk[c, c]
    if Y == 1:
        return normal.cdf(mu_j_given_Zk / sqrt(l ** -2 + sigma_j_given_Zk_sq))
    else:
        return 1 - normal.cdf(mu_j_given_Zk / sqrt(l ** -2 + sigma_j_given_Zk_sq))


def compute_E_Zi_given_Y_Zk_c(i, Y, Zk, c, Sigma_k, alpha, mu_zzk, Sigma_zzk):
    L = len(alpha)
    mu_i_k = mu_zzk[i]
    sigma_i_k_sq = Sigma_zzk[i, i]
    j = c
    S_11, S_12 = Sigma_k[:L, :L], Sigma_k[:L, L:]
    S_21, S_22 = Sigma_k[L:, :L], Sigma_k[L:, L:]
    j = c
    cov_Zj_Zi_Zk = np.block([
        [S_11[j, j], S_11[j, i], S_12[j, :]],
        [S_11[i, j], S_11[i, i], S_12[i, :]],
        [S_21[:, j, None], S_21[:, i, None], S_22]
    ])
    l = sqrt(np.pi / 8)
    sigma_j_ik_sq = cov_Zj_Zi_Zk[0, 0] - cov_Zj_Zi_Zk[0, 1:] @ pinv(cov_Zj_Zi_Zk[1:, 1:]) @ cov_Zj_Zi_Zk[1:, 0]
    mid = cov_Zj_Zi_Zk[0, 1:] @ pinv(cov_Zj_Zi_Zk[1:, 1:])
    alpha_j_i = mid[0] / sqrt(l ** -2 + sigma_j_ik_sq)
    gamma_j_i = mid[1:] @ Zk / sqrt(l ** -2 + sigma_j_ik_sq)
    eta_ijk = alpha_j_i * mu_i_k + gamma_j_i
    inner = eta_ijk / sqrt(1 + alpha_j_i ** 2 * sigma_i_k_sq)
    integral = (alpha_j_i * sigma_i_k_sq / sqrt(1 + alpha_j_i ** 2 * sigma_i_k_sq)) * \
        normal.pdf(inner) + mu_i_k * normal.cdf(inner)
    P_Y_1_given_Zk_c = compute_P_Y_given_Zk_c(1, j, mu_zzk, Sigma_zzk)
    if Y == 1:
        return integral / P_Y_1_given_Zk_c
    elif Y == 0:
        return (mu_i_k - integral) / (1 - P_Y_1_given_Zk_c)


def compute_E_Z_given_Zk_Y(Zk, Y, Sigma_k, alpha):
    L = len(alpha)
    mu_zzk = Sigma_k[:L, L:] @ pinv(Sigma_k[L:, L:]) @ Zk
    Sigma_zzk = Sigma_k[:L, :L] - Sigma_k[:L, L:] @ pinv(Sigma_k[L:, L:]) @ Sigma_k[L:, :L]
    E_Z_given_Zk_Y = np.zeros_like(alpha)
    P_c_given_Zk_Y = compute_P_c_given_Zk_Y(Y, alpha, mu_zzk, Sigma_zzk)
    for j in range(L):
        E_Z_given_Y_Zk_c = np.array([compute_E_Zi_given_Y_Zk_c(
            i, Y, Zk, j, Sigma_k, alpha, mu_zzk, Sigma_zzk) for i in range(L)])
        E_Z_given_Zk_Y += E_Z_given_Y_Zk_c * P_c_given_Zk_Y[j]
    return E_Z_given_Zk_Y
