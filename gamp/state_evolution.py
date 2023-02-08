import numpy as np


def state_evolution_mse(mu_k, sigma_k_sq, sigma_beta_sq):
    """ Prediction of the mean squared error from the state evolution, assuming a Gaussian prior. """
    q = mu_k * sigma_beta_sq / (mu_k ** 2 * sigma_beta_sq + sigma_k_sq)
    return sigma_beta_sq - q * mu_k * sigma_beta_sq
