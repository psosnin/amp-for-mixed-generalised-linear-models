from math import sqrt

import numpy as np
from numpy.linalg import inv, pinv
from scipy.stats import norm as normal


def compute_gk_1d(Z_k_Ybar, S_11, S_12, S_21, S_22, cov_Z_given_Zk, L, sigma_sq, alpha):
    """ Compute the Bayesian-Optimal g_k* """
    E_Z_given_Zk = S_12 @ pinv(S_22) @ Z_k_Ybar[0:L]
    E_Z_given_Zk_Ybar = compute_E_Z_given_Zk_Y(Z_k_Ybar, S_11, S_12, S_21, S_22, L, sigma_sq, alpha)
    g_Zk_Ybar = inv(cov_Z_given_Zk) @ (E_Z_given_Zk_Ybar - E_Z_given_Zk)
    return g_Zk_Ybar


def apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Apply Bayesian-Optimal g_k* to each row of Theta_k """
    cov_Z_given_Zk = S_11 - S_12 @ pinv(S_22) @ S_21
    Theta_k_Ybar = np.hstack((Theta_k, Y))
    return np.apply_along_axis(compute_gk_1d, 1, Theta_k_Ybar, S_11, S_12, S_21, S_22, cov_Z_given_Zk, L, sigma_sq, alpha)


def compute_E_Z_given_Zk_Y(Z_k_Ybar, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
    """ Compute E[Z | Z_k, Y] """
    Y, Zk = Z_k_Ybar[-1], Z_k_Ybar[:-1]
    E_Z_given_Zk_Y = np.zeros(L)
    E_Z_given_Zk = S_12 @ inv(S_22) @ Zk  # vector of mu_j's
    var_Z_given_Zk = np.diag(S_11 - S_12 @ inv(S_22) @ S_21)  # vector of sigma_j_sq's
    for c in range(L):
        P_c = compute_P_c_given_Zk_Y(c, Y, L, alpha, E_Z_given_Zk, var_Z_given_Zk, sigma_sq)
        E_c = compute_E_Z_given_Zk_Y_cj(c, Zk, Y, E_Z_given_Zk, var_Z_given_Zk, S_11, S_12, S_21, S_22, sigma_sq)
        E_Z_given_Zk_Y += P_c * E_c
    return E_Z_given_Zk_Y


def compute_E_Z_given_Zk_Y_cj(j, Zk, Y, E_Z_given_Zk, var_Z_given_Zk, S_11, S_12, S_21, S_22, sigma_sq):
    """ Compute E[Z | Z_k, Y, c = j] """
    L = len(Zk)
    E_0 = np.array([compute_E_Zi_given_Zk_Y_cj_omegaj_0(i, j, Zk, E_Z_given_Zk,
                   var_Z_given_Zk, S_11, S_12, S_21, S_22) for i in range(L)])
    E_1 = np.array([compute_E_Zi_given_Zk_Y_cj_omegaj_1(i, j, Zk, Y, E_Z_given_Zk,
                   var_Z_given_Zk, S_11, S_12, S_21, S_22, sigma_sq) for i in range(L)])
    P_0 = compute_P_omegaj_given_Zk_Y_c(0, Y, j, E_Z_given_Zk, var_Z_given_Zk, sigma_sq)
    P_1 = compute_P_omegaj_given_Zk_Y_c(1, Y, j, E_Z_given_Zk, var_Z_given_Zk, sigma_sq)
    return E_0 * P_0 + E_1 * P_1


def compute_E_Zi_given_Zk_Y_cj_omegaj_1(i, j, Zk, Y, E_Z_given_Zk, var_Z_given_Zk, S_11, S_12, S_21, S_22, sigma_sq):
    """ Compute E[Z_i | Z_k, Y, c = j, omega_j = 1] """
    S_a = np.hstack((S_11[j, i], S_12[j, :]))
    S_ba = np.vstack((
        np.hstack([S_11[i, i], S_12[i, :]]),
        np.hstack([S_21[:, i, None], S_22])
    ))
    mu_j, mu_i = E_Z_given_Zk[j], E_Z_given_Zk[i]
    sigma_j_sq, sigma_i_sq = var_Z_given_Zk[j], var_Z_given_Zk[i]
    sigma_ji_sq = S_11[j, j] - S_a @ inv(S_ba) @ S_a.T + 1e-10
    mid = S_a @ inv(S_ba)
    a_ji = mid[0]
    b_ji = mid[1:] @ Zk
    c_ji = (a_ji * sigma_sq) / (sigma_sq + sigma_ji_sq)
    d_ji = (Y * sigma_ji_sq + b_ji * sigma_sq) / (sigma_sq + sigma_ji_sq)
    s_ji_sq = sigma_sq * sigma_ji_sq / (sigma_sq + sigma_ji_sq)
    r_ji_sq = 1 / (1 / sigma_i_sq + a_ji ** 2 / sigma_ji_sq - c_ji ** 2 / s_ji_sq)
    w_ji = r_ji_sq * (mu_i / sigma_i_sq - a_ji * b_ji / sigma_ji_sq + c_ji * d_ji / s_ji_sq)

    # add first then take exp to avoid overflow
    D_ji = Y ** 2 / (2 * sigma_sq) + mu_i ** 2 / (2 * sigma_i_sq) + b_ji ** 2 / (2 * sigma_ji_sq)
    D_ji += - d_ji ** 2 / (2 * s_ji_sq) - w_ji ** 2 / (2 * r_ji_sq)
    D_ji = np.exp(D_ji)
    D_ji *= sqrt(sigma_i_sq / r_ji_sq)
    D_ji *= sqrt(2 * np.pi * (sigma_sq + sigma_ji_sq))
    D_ji *= (1 - normal.cdf(0, mu_j, sqrt(sigma_j_sq)))
    D_ji *= compute_P_Y_given_omegaj_Zk_c(Y, 1, j, E_Z_given_Zk, var_Z_given_Zk, sigma_sq)
    D_ji += 1e-10

    gamma = 1 / np.sqrt(s_ji_sq + c_ji ** 2 * r_ji_sq)
    arg1 = gamma * (- c_ji * w_ji - d_ji)
    arg2 = gamma * c_ji * r_ji_sq
    return (w_ji + arg2 * normal.pdf(arg1) - w_ji * normal.cdf(arg1)) / D_ji


def compute_E_Zi_given_Zk_Y_cj_omegaj_0(i, j, Zk, E_Z_given_Zk, var_Z_given_Zk, S_11, S_12, S_21, S_22):
    """ Compute E[Z_i | Z_k, Y, c = j, omega_j = 0] """
    S_a = np.hstack((S_11[j, i], S_12[j, :]))
    S_ba = np.vstack((
        np.hstack([S_11[i, i], S_12[i, :]]),
        np.hstack([S_21[:, i, None], S_22])
    ))
    mu_j, mu_i = E_Z_given_Zk[j], E_Z_given_Zk[i]
    sigma_j_sq, sigma_i_sq = var_Z_given_Zk[j], var_Z_given_Zk[i]
    sigma_ji_sq = S_11[j, j] - S_a @ inv(S_ba) @ S_a.T
    mid = S_a @ inv(S_ba)
    a_ji = mid[0]
    b_ji = mid[1:] @ Zk
    gamma = 1 / (sqrt(sigma_ji_sq + a_ji ** 2 * sigma_i_sq))
    arg1 = gamma * (- a_ji * mu_i - b_ji)
    arg2 = gamma * (- sigma_i_sq * a_ji)
    return (arg2 * normal.pdf(arg1) + mu_i * normal.cdf(arg1)) / normal.cdf(0, mu_j, sqrt(sigma_j_sq))


def compute_P_Y_given_omegaj_Zk_c(Y, omega_j, c, E_Z_given_Zk, var_Z_given_Zk, sigma_sq):
    """ Compute P(Y | omega_j, Z_k, cbar = c) """
    if omega_j == 0:
        return normal.pdf(Y, 0, sqrt(sigma_sq))
    elif omega_j == 1:
        mu_j = E_Z_given_Zk[c]
        sigma_j_sq = var_Z_given_Zk[c]
        m_j = (Y * sigma_j_sq + mu_j * sigma_sq) / (sigma_sq + sigma_j_sq)
        s_sq_j = sigma_sq * sigma_j_sq / (sigma_sq + sigma_j_sq)

        C_j = sqrt(2 * np.pi * (sigma_sq + sigma_j_sq))
        C_j *= (1 - normal.cdf(0, mu_j, sqrt(sigma_j_sq)))
        C_j *= np.exp(mu_j ** 2 / (2 * sigma_j_sq))
        C_j += 1e-10

        result = np.exp(- Y ** 2 / (2 * sigma_sq) + m_j ** 2 / (2 * s_sq_j))
        result *= (1 - normal.cdf(0, m_j, sqrt(s_sq_j)))
        result /= C_j
        return result


def compute_P_Y_given_Zk_c(Y, c, sigma_sq, E_Z_given_Zk, var_Z_given_Zk):
    """ Compute P(Y | Z_k, cbar = c) """
    mu_j, sigma_j_sq = E_Z_given_Zk[c], var_Z_given_Zk[c]
    P_omegaj_0_given_Zk_c = normal.cdf(0, mu_j, sqrt(sigma_j_sq))
    P_omegaj_1_given_Zk_c = 1 - P_omegaj_0_given_Zk_c
    P_Y_given_omegaj_0_Zk_c = compute_P_Y_given_omegaj_Zk_c(Y, 0, c, E_Z_given_Zk, var_Z_given_Zk, sigma_sq)
    P_Y_given_omegaj_1_Zk_c = compute_P_Y_given_omegaj_Zk_c(Y, 1, c, E_Z_given_Zk, var_Z_given_Zk, sigma_sq)
    return P_omegaj_0_given_Zk_c * P_Y_given_omegaj_0_Zk_c + P_omegaj_1_given_Zk_c * P_Y_given_omegaj_1_Zk_c


def compute_P_c_given_Zk_Y(c, Y, L, alpha, E_Z_given_Zk, var_Z_given_Zk, sigma_sq):
    """ Compute p(c | Z_k, Y) """
    P_Y_given_Zk_c = np.array([compute_P_Y_given_Zk_c(Y, j, sigma_sq, E_Z_given_Zk, var_Z_given_Zk) for j in range(L)])
    r = alpha[c] * P_Y_given_Zk_c[c] / np.sum(alpha[:] * P_Y_given_Zk_c)
    return r


def compute_P_omegaj_given_Zk_Y_c(omega_j, Y, c, E_Z_given_Zk, var_Z_given_Zk, sigma_sq):
    """ Compute P(omega_j | Z_k, Y, cbar=c) """
    mu_j, sigma_j_sq = E_Z_given_Zk[c], var_Z_given_Zk[c]
    P_Y_given_Zk_c = compute_P_Y_given_Zk_c(Y, c, sigma_sq, E_Z_given_Zk, var_Z_given_Zk)
    if omega_j == 0:
        return normal.pdf(Y, 0, sqrt(sigma_sq)) * normal.cdf(0, mu_j, sqrt(sigma_j_sq)) / P_Y_given_Zk_c
    elif omega_j == 1:
        return 1 - normal.pdf(Y, 0, sqrt(sigma_sq)) * normal.cdf(0, mu_j, sqrt(sigma_j_sq)) / P_Y_given_Zk_c
