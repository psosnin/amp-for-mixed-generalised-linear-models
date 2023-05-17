import numpy as np
from numpy.linalg import pinv, norm


def matrix_GAMP(X, Y, B_hat_k, B_row_cov, sigma_sq, alpha, n_iters, apply_gk, eps=1e-2):
    """
    Run matrix generalised approximate message passing to estimate B from X and Y
    with a given initial estimate B_hat_k.
    The model makes the following assumptions:
        1. Y is generated from a mixed generalised linear model, Y = q(X @ B) + noise, where q is a link function.
        2. The rows of B are drawn iid from a zero mean gaussian distribution with covariance B_row_cov.
        3. The covariates X are drawn iid from a zero mean gaussian distribution with covariance 1 / n.
        4. The noise is drawn iid from a zero mean gaussian distribution with variance sigma_sq.
        5. Y_i is generated from a single column of B according to a categorical distribution with probabilities alpha.
        6. Initial estimate of B is drawn from the true prior distribution of B.
    Parameters:
        X: n x p = covariate matrix
        Y: n x 1 = observations
        B_hat_k: p x L = initial estimate of signal matrix B
        B_row_cov: L x L = prior covariance of distribution of the rows of B
        sigma: float = noise variance
        alpha: L x 1 = categorical distribution on components
        n_iters: int = max number of AMP iterations to perform
        gk_expect: function to compute E[Z | Zk, Ybar], depends on choice of GLM
        eps: float = stopping criterion
    Returns:
        B_hat_list: list of estimates of B for each AMP iteration
        M_k_B_list: list of estimates of M_K_B for each AMP iteration
    """
    # get dimensions and validate Y
    n, p = X.shape
    L = B_hat_k.shape[1]
    assert (Y.shape == (n, 1))
    # initialise lists to store estimates
    B_hat_list = [B_hat_k.copy()]
    M_k_B_list = []
    # initialise R_hat_minus_1, F_0, Sigma_0
    R_hat_minus_1 = np.zeros((n, L))
    F_k = np.eye(L)
    Sigma_k = p * np.block([[B_row_cov, np.zeros((L, L))], [np.zeros((L, L)), B_row_cov]]) / n

    # begin AMP iterations
    for _ in range(n_iters):
        # split Sigma_k into components
        S_1, S_2 = np.hsplit(Sigma_k, 2)
        S_11, S_21 = np.vsplit(S_1, 2)
        S_12, S_22 = np.vsplit(S_2, 2)
        # step 1: compute Theta_k
        Theta_k = X @ B_hat_k - R_hat_minus_1 @ F_k.T
        # step 2: compute R_hat_k using denoising function gk and check for nan
        R_hat_k = apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha)
        if np.isnan(R_hat_k).any():
            print(f"nan in R_hat_k at iteration {_}, stopping.")
            break
        # step 3: compute C_k
        C_k = compute_Ck(Theta_k, R_hat_k, S_21, S_22, n)
        # step 4: compute B_k estimate of B
        B_k_plus_1 = X.T @ R_hat_k - B_hat_k @ C_k.T
        # step 5: compute M_B_kand B_hat_k using denoising function fk
        M_B_k_plus_1 = R_hat_k.T @ R_hat_k / n
        B_hat_k_plus_1 = apply_fk(B_k_plus_1, M_B_k_plus_1, B_row_cov)
        # step 6: compute F_k_plus_1
        F_k_plus_1 = compute_Fk(M_B_k_plus_1, B_row_cov, p, n)
        # step 7: compute updated Sigma_k
        Sigma_k_plus_1 = update_Sigmak(B_hat_k_plus_1, B_row_cov, p, n)
        # store estimates
        B_hat_list.append(B_hat_k_plus_1.copy())
        M_k_B_list.append(M_B_k_plus_1)
        # check convergence criterion
        if norm(B_hat_k - B_hat_k_plus_1) ** 2 / p < eps:
            print(f"Convergence criterion met at iteration {_}")
            break
        # update variables for next iteration
        B_hat_k = B_hat_k_plus_1
        Sigma_k = Sigma_k_plus_1
        R_hat_minus_1 = R_hat_k
        F_k = F_k_plus_1

    return B_hat_list, M_k_B_list


def compute_Ck(Theta_k, R_hat_k, S_21, S_22, n):
    """ Approximate C_k = g_k'(Theta_k, Ybar) / n """
    return (pinv(S_22) @ (Theta_k.T @ R_hat_k - S_21 @ R_hat_k.T @ R_hat_k) / n).T


def compute_Fk(M_B_k, B_cov, p, n):
    """ Approximate F_k = f_k'(B) / n """
    return p * pinv(M_B_k @ B_cov @ M_B_k.T + M_B_k) @ M_B_k @ B_cov / n


def apply_fk(B_k, M_B_k, B_cov):
    """ Apply denoising function f_k to B_k, where we assume a zero mean gaussian prior with covariance B_cov """
    result = pinv(M_B_k @ B_cov @ M_B_k.T + M_B_k)
    result = B_k @ (B_cov @ M_B_k.T @ result).T
    return result


def update_Sigmak(B_hat_k, B_cov, p, n):
    """ Update state evolution Sigma_k using B_hat_k and B_cov """
    B_hat_cov = B_hat_k.T @ B_hat_k / p
    tmp = np.block([[B_cov, B_hat_cov], [B_hat_cov, B_hat_cov]])
    return p * tmp / n
