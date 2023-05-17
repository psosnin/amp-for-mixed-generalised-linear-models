import numpy as np
from numpy.linalg import pinv, norm, inv
from scipy.stats import multivariate_normal


"""
implement a vectorised version of the Bayes-optimal g_k* for the mixed linear regression model
"""


def apply_gk(Theta_k, Y, S_11, S_12, S_21, S_22, L, sigma_sq, alpha):
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
        P_Zk_Ybar_given_cbar = np.array(
            [multivariate_normal.pdf(Theta_k_Ybar, mean=np.zeros(L+1), cov=cov, allow_singular=True)
             for cov in cov_Zk_Ybar_given_cbar]
        )
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
