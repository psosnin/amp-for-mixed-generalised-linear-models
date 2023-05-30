from math import sqrt

import numpy as np
from numpy.linalg import norm, pinv

from .generate_data import sigmoid

"""
file containing loss functions
"""


def mse(beta, beta_hat, sigma_beta_sq=None):
    """
    Mean squared error between signal and estimate normalised by the dimension.
    """
    if sigma_beta_sq:
        return norm(beta - beta_hat) ** 2 / (sigma_beta_sq * beta.shape[0])
    return norm(beta - beta_hat) ** 2 / beta.shape[0]


def state_evolution_mse(mu_k, sigma_k_sq, sigma_beta_sq, normalise=False):
    """ Prediction of the mean squared error from the state evolution, assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    if normalise:
        return 1 - q * mu_k
    return sigma_beta_sq - q * mu_k * sigma_beta_sq


def mse_mixed(B_hat, B, B_cov=None, permute=False):
    """
    Find the sum of the mean squared error between the true regression slopes and their estimates normalised by the
    dimension.
    If permute is True, then the MSE is calculated by finding the permutation of the estimated slopes that minimises
    the MSE.
    Parameters:
        B_hat: p x L = estimate of signal matrix
        B: p x L = true signal matrix
        B_cov: p x p = covariance matrix of B, if provided, MSE is normalised by diagonal of B_cov
        permute: bool = whether to consider permutations of B_hat matrix
    Returns:
        mse: mean squared error of estimated signal and true signal
    """
    p, L = B_hat.shape
    if not permute:
        mse = (norm(B - B_hat, axis=0) ** 2 / p)
    else:
        mse = np.repeat(B_hat[:, None].T, L, axis=1)
        mse = norm(mse - B.T, axis=2) ** 2 / p
        mse_list = np.zeros(L)
        for j in range(L):
            i = np.argmin(mse[:, j])
            mse_list[j] = mse[i, j]
            mse[i, :] = np.inf
        mse = mse_list
    return mse if B_cov is None else mse / np.diag(B_cov)


def state_evolution_mse_mixed(M_k, Sigma_B, normalise=False):
    L = M_k.shape[0]
    Q = Sigma_B @ M_k.T @ pinv(M_k @ Sigma_B @ M_k.T + M_k.T)
    se = []
    for i in range(L):
        se.append(Sigma_B[i, i] - Q[i, :] @ M_k @ Sigma_B[i, :])
    if normalise:
        se = [se[i] / Sigma_B[i, i] for i in range(L)]
    return se


def norm_sq_corr(beta, beta_hat):
    """
    Calculate the normalised squared correlation between beta and beta_hat.
    """
    return (np.dot(beta, beta_hat) / (norm(beta) * norm(beta_hat))) ** 2


def state_evolution_corr(mu_k, sigma_k_sq, sigma_beta_sq):
    """ Prediction of the normalised correlation from the state evolution assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    return q * mu_k


def prediction_error(beta, beta_hat, n, RNG, scale=10):
    """
    Calculate the prediction error of the model with signal beta_hat vs the true signal beta by simulating data.
    """
    p = beta.size
    X = RNG.normal(0, sqrt(1 / n), (scale*n, p))
    # data
    u = RNG.uniform(0, 1, scale*n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)

    # predictions:
    y_pred_hat = np.array(sigmoid(X @ beta_hat) > 0.5, dtype=int)
    return (scale*n - np.sum(y_pred_hat == y)) / (scale*n)


def prediction_error_mixed(B, B_hat, n, RNG, scale=10, combined=False, alpha=None):
    """
    Calculate the prediction error of the model with signal B_hat vs the true signal B by simulating data.
    """
    if not combined:
        return np.array([prediction_error(B[:, j], B_hat[:, j], n, RNG, scale) for j in range(B.shape[1])])
    p, L = B.shape
    X = RNG.normal(0, np.sqrt(1 / n), (scale*n, p))
    Theta = X @ B
    # generate class label latent variables from Cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, scale*n), np.cumsum(alpha))

    # generate Y by picking elements from Theta according to c
    Y = np.take_along_axis(Theta, c[:, None], axis=1)
    u = RNG.uniform(0, 1, (scale*n, 1))
    Y = np.array(sigmoid(Y) > u, dtype=int).flatten()

    Y_pred = np.array(np.sum(alpha * sigmoid(X @ B_hat), axis=1) > 0.5, dtype=int).flatten()
    return (scale*n - np.sum(Y_pred == Y)) / (scale*n)


def state_evolution_corr_mixed(M_k, Sigma_B):
    """
    State evolution prediction of the normalised correlation between the true signal and the estimated signal.
    """
    L = M_k.shape[0]
    Q = Sigma_B @ M_k.T @ pinv(M_k @ Sigma_B @ M_k.T + M_k.T)
    se = []
    for i in range(L):
        se.append(Q[i, :] @ M_k @ Sigma_B[i, :] / Sigma_B[i, i])
    return se


def norm_sq_corr_mixed(B_hat, B, permute=False):
    """
    Find the normalised squared correlation between the true regression slopes and their estimates.
    Parameters:
        B_hat: p x L = estimate of signal matrix
        B: p x L = true signal matrix
    Returns:
        corr: normalised squared correlation between estimated signal and true signal
    """
    if permute:
        p1 = [norm_sq_corr(B[:, j], B_hat[:, j]) for j in range(B.shape[1])]
        p2 = [norm_sq_corr(B[:, j], B_hat[:, -j-1]) for j in range(B.shape[1])]
        print(p1)
        print(p2)
        if np.sum(p2) > np.sum(p1):
            return p2
        else:
            return p1

    return [norm_sq_corr(B[:, j], B_hat[:, j]) for j in range(B.shape[1])]
