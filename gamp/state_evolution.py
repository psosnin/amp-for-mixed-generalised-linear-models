import numpy as np
from numpy.linalg import pinv


def state_evolution_mse(mu_k, sigma_k_sq, sigma_beta_sq):
    """ Prediction of the mean squared error from the state evolution, assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    return sigma_beta_sq - q * mu_k * sigma_beta_sq


def state_evolution_corr(mu_k, sigma_k_sq, sigma_beta_sq):
    """ Prediction of the normalised correlation from the state evolution assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    return q * mu_k

def state_evolution_mse_mixed(M_k, Sigma_B):
    L = M_k.shape[0]
    Q = Sigma_B @ M_k.T @ pinv(M_k @ Sigma_B @ M_k.T + M_k.T)
    se = []
    for i in range(L):
        se.append(Sigma_B[i, i] - Q[i, :] @ M_k @ Sigma_B[i, :])
    return se
