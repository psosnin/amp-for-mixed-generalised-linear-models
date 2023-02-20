import numpy as np
from tqdm import tqdm
from numpy.linalg import inv, pinv


def matrix_GAMP(X, Y, B_hat_k, B_row_cov, sigma_sq, alpha, n_iters, apply_gk):
    """
    Run matrix generalised approximate message passing to estimate B from X and Y.
    Parameters:
        X: n x p = samples
        Y: n x 1 = observations
        B_hat_k: p x L = initial estimate of signal matrix
        B_row_cov: L x L = covariance of distribution of the rows of B
        sigma: int = noise variance
        alpha: L x 1 = categorical distribution on components
        n_iters: int = max number of AMP iterations to perform
        gk_expect: function to compute E[Z | Zk, Ybar], depends on choice of GLM
    Returns:
        B_hat_list: list of estimates of B for each AMP iteration
        M_k_B_list: list of estimates of M_K_B for each AMP iteration
    """
    n, p = X.shape
    L = B_hat_k.shape[1]
    assert (Y.shape == (n, 1))
    B_hat_list = [B_hat_k.copy()]
    M_k_B_list = []
    # initialise R_hat_minus_1, F_0, Sigma_0
    R_hat_minus_1 = np.zeros((n, L))
    F_k = np.eye(L)
    # TODO: make this work if the mean is not zero and
    # if B and B_hat_0 have different distributions
    Sigma_k = p * np.block([[B_row_cov, np.zeros((L, L))],
                           [np.zeros((L, L)), B_row_cov]]) / n

    # begin AMP iterations
    for _ in tqdm(range(n_iters), disable=True):
        # print(Sigma_k)
        S_1, S_2 = np.hsplit(Sigma_k, 2)
        S_11, S_21 = np.vsplit(S_1, 2)
        S_12, S_22 = np.vsplit(S_2, 2)
        # step 1:
        Theta_k = X @ B_hat_k - R_hat_minus_1 @ F_k.T
        # step 2:
        R_hat_k = apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha)
        if np.isnan(R_hat_k).any():
            print(f"nan in R_hat_k at iteration {_}, stopping.")
            break
        # step 3:
        C_k = compute_Ck(Theta_k, R_hat_k, S_21, S_22, n)
        # step 4:
        B_k_plus_1 = X.T @ R_hat_k - B_hat_k @ C_k.T
        # step 5:
        M_B_k_plus_1 = R_hat_k.T @ R_hat_k / n
        B_hat_k_plus_1 = apply_fk(B_k_plus_1, M_B_k_plus_1, B_row_cov)

        # step 6:
        F_k_plus_1 = compute_Fk(M_B_k_plus_1, B_row_cov, p, n)
        # step 7:
        Sigma_k_plus_1 = update_Sigmak(B_hat_k_plus_1, B_row_cov, p, n)

        # prepare next iteration:
        B_hat_k = B_hat_k_plus_1
        Sigma_k = Sigma_k_plus_1
        R_hat_minus_1 = R_hat_k
        F_k = F_k_plus_1
        B_hat_list.append(B_hat_k.copy())
        M_k_B_list.append(M_B_k_plus_1)
    return B_hat_list, M_k_B_list


def compute_Ck(Theta_k, R_hat_k, S_21, S_22, n):
    return (pinv(S_22) @ (Theta_k.T @ R_hat_k - S_21 @ R_hat_k.T @ R_hat_k) / n).T


def compute_Fk(M_B_k, B_cov, p, n):
    return p * pinv(M_B_k @ B_cov @ M_B_k.T + M_B_k) @ M_B_k @ B_cov / n


def apply_fk(B_k, M_B_k, B_cov):
    result = pinv(M_B_k @ B_cov @ M_B_k.T + M_B_k)
    result = B_k @ (B_cov @ M_B_k.T @ result).T
    return result


def update_Sigmak(B_hat_k, B_cov, p, n):
    B_hat_cov = B_hat_k.T @ B_hat_k / p
    tmp = np.block([[B_cov, B_hat_cov],
                    [B_hat_cov, B_hat_cov]])
    return p * tmp / n
