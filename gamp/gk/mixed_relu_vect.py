from math import sqrt

import numpy as np
from numpy.linalg import inv, pinv
from scipy.stats import norm as normal

"""
Implement a vectorised version of the Bayes-optimal g_k* for the mixed logistic regression model.
Runs much faster than the original version but is less readable.
"""


def apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Apply Bayesian-Optimal g_k* to each row of Theta_k """
    cov_Z_given_Zk = S_11 - S_12 @ pinv(S_22) @ S_21
    E_Z_given_Zk = (S_12 @ pinv(S_22) @ Theta_k.T).T
    E_Z_given_Zk_Ybar = compute_E_Z_given_Zk_Y(Theta_k, Y, E_Z_given_Zk, sigma_sq, alpha, S_11, S_12, S_21, S_22)
    return (E_Z_given_Zk_Ybar - E_Z_given_Zk) @ inv(cov_Z_given_Zk)


def compute_E_Z_given_Zk_Y(Theta_k, Y, E_Z_given_Zk, sigma_sq, alpha, S_11, S_12, S_21, S_22):
    """ Compute E[Z | Zk, Y] """
    # compute covariances
    cov_Zj_Zi_Zk = compute_cov_Zj_Zi_Zk(S_11, S_12, S_21, S_22)
    var_Z_given_Zk = np.clip(np.diag(S_11 - S_12 @ pinv(S_22) @ S_21), 0, np.inf) + 1e-10  # for numerical stability
    cov_Zj_given_Zi_Zk = compute_cov_Zj_given_Zi_Zk(cov_Zj_Zi_Zk)

    # compute constants
    ac, bc, cc, dc = get_constants(cov_Zj_Zi_Zk, Theta_k, Y, cov_Zj_given_Zi_Zk, sigma_sq)

    # compute intermediate probabilities
    P_omegaj_0_given_Zk_c = normal.cdf(0, E_Z_given_Zk, np.sqrt(var_Z_given_Zk))
    P_omegaj_1_given_Zk_c = 1 - P_omegaj_0_given_Zk_c
    P_Y_given_omegaj_0_Zk_c = normal.pdf(Y, 0, sqrt(sigma_sq))
    P_Y_given_omegaj_1_Zk_c = compute_P_Y_given_omega_1_Zk_c(Y, sigma_sq, E_Z_given_Zk, var_Z_given_Zk)
    P_Y_given_Zk_c = P_omegaj_0_given_Zk_c * P_Y_given_omegaj_0_Zk_c + P_omegaj_1_given_Zk_c * P_Y_given_omegaj_1_Zk_c
    P_c_given_Zk_Y = alpha * P_Y_given_Zk_c / np.sum(alpha * P_Y_given_Zk_c, axis=1)[:, None]
    P_omegaj_0_given_Zk_Y_c = P_Y_given_omegaj_0_Zk_c * P_omegaj_0_given_Zk_c / P_Y_given_Zk_c
    P_omegaj_1_given_Zk_Y_c = 1 - P_omegaj_0_given_Zk_Y_c

    # compute intermediate expectations
    E_Z_given_Zk_Y_c_omega_0 = compute_E_Z_given_Zk_Y_c_omega_0(
        E_Z_given_Zk, var_Z_given_Zk, cov_Zj_given_Zi_Zk, ac, bc)
    E_Z_given_Zk_Y_c_omega_1 = compute_E_Z_given_Zk_Y_c_omega_1(
        Y, E_Z_given_Zk, var_Z_given_Zk, cov_Zj_given_Zi_Zk, ac, bc, cc, dc, sigma_sq, P_Y_given_omegaj_1_Zk_c)
    E_Z_given_Zk_Ybar_c = (
        E_Z_given_Zk_Y_c_omega_0 * P_omegaj_0_given_Zk_Y_c[:, :, None] +
        E_Z_given_Zk_Y_c_omega_1 * P_omegaj_1_given_Zk_Y_c[:, :, None]
    )

    # compute E[Z | Zk, Ybar]
    E_Z_given_Zk_Ybar = np.sum(E_Z_given_Zk_Ybar_c * P_c_given_Zk_Y[:, :, None], axis=1)
    return E_Z_given_Zk_Ybar


def compute_E_Z_given_Zk_Y_c_omega_1(Y, E_Z_given_Zk, var_Z_given_Zk, cov_Zj_given_Zi_Zk, ac, bc, cc, dc, sigma_sq, P_Y_given_omegaj_1_Zk_c):
    """ Compute E[Zi | Z_k, Y, c = j, omega_j = 1]. Returns n x L x L matrix indexed by n, j, i"""
    # compute w_ji, r_ji_sq and s_ji_sq
    s_sq = sigma_sq * cov_Zj_given_Zi_Zk / (sigma_sq + cov_Zj_given_Zi_Zk)
    r_sq = 1 / (1 / var_Z_given_Zk + ac ** 2 / cov_Zj_given_Zi_Zk - cc ** 2 / s_sq)
    w = r_sq * (E_Z_given_Zk[:, None] / var_Z_given_Zk
                - ac[None, :, :] * bc / cov_Zj_given_Zi_Zk[None, :, :]
                + cc[None, :, :] * dc / s_sq[None, :, :])

    # compute normalisation constant. add first then take exp to avoid overflow
    D_ji = bc ** 2 / (2 * cov_Zj_given_Zi_Zk)
    D_ji += - dc ** 2 / (2 * s_sq)
    D_ji += - w ** 2 / (2 * r_sq)
    D_ji += Y[:, None] ** 2 / (2 * sigma_sq)
    D_ji += E_Z_given_Zk[:, None] ** 2 / (2 * var_Z_given_Zk)
    D_ji = np.exp(D_ji)
    D_ji *= np.sqrt(var_Z_given_Zk / r_sq)
    D_ji *= np.sqrt(2 * np.pi * (sigma_sq + cov_Zj_given_Zi_Zk))
    D_ji *= (1 - normal.cdf(0, E_Z_given_Zk, np.sqrt(var_Z_given_Zk)))[:, :, None]
    D_ji *= P_Y_given_omegaj_1_Zk_c[:, :, None]
    D_ji += 1e-10  # avoid division by zero

    gamma = 1 / np.sqrt(s_sq + cc ** 2 * r_sq)
    arg1 = gamma * (- cc * w - dc)
    arg2 = gamma * cc * r_sq
    return (w + arg2 * normal.pdf(arg1) - w * normal.cdf(arg1)) / D_ji


def compute_P_Y_given_omega_1_Zk_c(Y, sigma_sq, E_Z_given_Zk, var_Z_given_Zk):
    """ Compute P[Y | omega_j = 1, Z_k, c]. Returns an n x L matrix indexed by n, j """
    s_sq = (sigma_sq * var_Z_given_Zk) / (sigma_sq + var_Z_given_Zk)
    m = (Y * var_Z_given_Zk + E_Z_given_Zk * sigma_sq) / (sigma_sq + var_Z_given_Zk)
    C_j = (1 - normal.cdf(0, E_Z_given_Zk, np.sqrt(var_Z_given_Zk)))
    C_j *= np.sqrt(2 * np.pi * (sigma_sq + var_Z_given_Zk))
    C_j += 1e-10
    # add then take exp to avoid overflow (exp part of the normalisation constant C_j is also included here)
    exponent = - Y ** 2 / (2 * sigma_sq) + m ** 2 / (2 * s_sq) - E_Z_given_Zk ** 2 / (2 * var_Z_given_Zk)
    P_Y_given_omegaj_1_Zk_c = np.exp(exponent)
    P_Y_given_omegaj_1_Zk_c *= (1 - normal.cdf(0, m, np.sqrt(s_sq))) / C_j
    return P_Y_given_omegaj_1_Zk_c


def compute_E_Z_given_Zk_Y_c_omega_0(E_Z_given_Zk, var_Z_given_Zk, cov_Zj_given_Zi_Zk, ac, bc):
    """ Compute E[Zi | Z_k, Y, c = j, omega_j = 0]. Returns an n x L x L matrix indexed by (n, j, i) """
    gamma = 1 / np.sqrt(cov_Zj_given_Zi_Zk + ac ** 2 * var_Z_given_Zk)
    arg1 = - gamma * (ac * E_Z_given_Zk[:, None] + bc)
    arg2 = - gamma * ac * var_Z_given_Zk
    numerator = arg2 * normal.pdf(arg1) + E_Z_given_Zk[:, None] * normal.cdf(arg1)
    return numerator / normal.cdf(0, E_Z_given_Zk, np.sqrt(var_Z_given_Zk))[:, :, None]


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
    cov_Zj_given_Zi_Zk = np.clip(cov_Zj_given_Zi_Zk, 0, np.inf) + 1e-10  # avoid numerical issues
    return cov_Zj_given_Zi_Zk


def get_constants(cov_Zj_Zi_Zk, Zk, Y, cov_Zj_given_Zi_Zk, sigma_sq):
    """
    Compute constants a, b, c, d such that
        a_ji * Z_i + b_ji =  E[Z_j | Z_i, Zk] = mu_ji
        c_ji = (a_ji * sigma_sq) / (sigma_sq + sigma_ji_sq)
        d_ji = (Y * sigma_ji_sq + b_ji * sigma_sq) / (sigma_sq + sigma_ji_sq)
    Parameters:
        cov_Zj_Zi_Zk: L x L x (L + 2) x (L + 2) matrix
        cov_Zj_given_Zi_Zk: L x L matrix
        Zk: n x L matrix, rows are individual Zk
        Y: n x 1 matrix, rows are individual Y
        E_Z_given_Zk: L x 1 matrix
        sigma_sq: scalar
    Returns:
        ac = L x L matrix indexed by j, i
        bc = n x L x L matrix indexed by n, j, i
        cc = L x L matrix indexed by j, i
        dc = n x L x L matrix indexed by n, j, i
    """
    n, L = Zk.shape
    ac = np.zeros((L, L))
    bc = np.zeros((n, L, L))
    dc = np.zeros((n, L, L))
    for j in range(L):
        const = (pinv(cov_Zj_Zi_Zk[j, :, 1:, 1:]) @ cov_Zj_Zi_Zk[j, :, 0, 1:, None])[:, :, 0]
        ac[j, :] = const[:, 0]
        bc[:, j, :] = ((const[:, 1:] @ Zk[:, :, None]))[:, :, 0]
        dc[:, j, :] = (Y * cov_Zj_given_Zi_Zk[j, :] + bc[:, j, :] * sigma_sq) / (sigma_sq + cov_Zj_given_Zi_Zk[j, :])
    cc = ac * sigma_sq / (sigma_sq + cov_Zj_given_Zi_Zk)
    return ac, bc, cc, dc
