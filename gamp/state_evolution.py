import numpy as np


def state_evolution_mse(mu_k, sigma_k_sq, sigma_beta_sq):
    """ Prediction of the mean squared error from the state evolution, assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    return sigma_beta_sq - q * mu_k * sigma_beta_sq


def state_evolution_corr(mu_k, sigma_k_sq, sigma_beta_sq):
    """ Prediction of the normalised correlation from the state evolution assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    return q * mu_k
