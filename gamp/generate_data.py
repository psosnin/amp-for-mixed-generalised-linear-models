from math import sqrt
import numpy as np
from numpy.random import default_rng


"""
Code for generating data according to the following generalised linear models:
    - linear regression
    - logistic regression
    - ReLU regression
    - mixed linear regression
    - mixed logistic regression
    - mixed ReLU regression
"""


def generate_data_linear(p, n, sigma_sq, sigma_beta_sq, RNG=default_rng()):
    """
    Generate a random dataset according to the linear regression model.
    Parameters:
        p: int = number of dimensions
        n: int = number of samples
        sigma_sq: float = noise variance
        sigma_beta_sq: float = signal variance
        RNG: numpy random number generator
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        beta: p x 1 = signal
        beta_hat: p x 1 = initial estimate of signal
    """
    beta = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    eps = RNG.normal(0, sqrt(sigma_sq), n)
    y = X @ beta + eps
    return X, y, beta, beta_hat_k


def sigmoid(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))


def generate_data_logistic(p, n, sigma_sq, sigma_beta_sq, RNG=default_rng()):
    """
    Generate a random dataset according to the logistic regression model.
    Parameters:
        p: int = number of dimensions
        n: int = number of samples
        sigma_beta_sq: float = signal variance
        RNG: numpy random number generator
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        beta: p x 1 = signal
        beta_hat: p x 1 = initial estimate of signal
    """
    beta = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    u = RNG.uniform(0, 1, n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)
    return X, y, beta, beta_hat_k


def generate_data_relu(p, n, sigma_sq, sigma_beta_sq, RNG=default_rng()):
    """
    Generate a random dataset according to the rectified linear regression model.
    Parameters:
        p: int = number of dimensions
        n: int = number of samples
        sigma_beta_sq: float = signal variance
        RNG: numpy random number generator
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        beta: p x 1 = signal
        beta_hat: p x 1 = initial estimate of signal
    """
    beta = RNG.normal(0, sqrt(sigma_beta_sq), p)
    beta_hat_k = RNG.normal(0, sqrt(sigma_beta_sq), p)
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    eps = RNG.normal(0, sqrt(sigma_sq), n)
    y = np.clip(X @ beta, 0, np.inf) + eps
    return X, y, beta, beta_hat_k


def generate_data_mixed_linear(p, L, n, alpha, B_row_cov, sigma_sq, RNG=default_rng()):
    """
    Generate a random dataset according to the mixed linear regression model.
    Parameters:
        p: int = number of dimensions
        L: int = number of mixture components
        n: int = number of samples
        alpha: L x 1 = categorical distribution on components
        B_row_cov: L x L = covariance matrix of the rows of B
        sigma_sq: float = noise variance
        RNG: numpy random number generator
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        B: p x L = signal matrix
        B_hat_0: p x L = initialisation of signal matrix estimate
    """
    # generate rows of signal matrix B according to zero mean Gaussian with covariance B_row_cov
    B = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    B_hat_0 = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    X = RNG.normal(0, np.sqrt(1 / n), (n, p))
    Theta = X @ B
    # generate class label latent variables from cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, n), np.cumsum(alpha))
    # generate Y by picking elements from Theta according to c
    Y = np.take_along_axis(Theta, c[:, None], axis=1)
    Y = Y + RNG.normal(0, sqrt(sigma_sq), n)[:, None]
    return X, Y, B, B_hat_0


def generate_data_mixed_logistic(p, L, n, alpha, B_row_cov, sigma_sq, RNG=default_rng()):
    """
    Generate a random dataset according to the mixed logistic regression model.
    Parameters:
        p: int = number of dimensions
        L: int = number of mixture components
        n: int = number of samples
        alpha: L x 1 = categorical distribution on components
        B_row_cov: L x L = covariance matrix of the rows of B
        sigma_sq: float = noise variance
        RNG: numpy random number generator
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        B: p x L = signal matrix
        B_hat_0: p x L = initialisation of signal matrix estimate
    """
    # generate rows of signal matrix B according to zero mean Gaussian with covariance B_row_cov
    B = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    B_hat_0 = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    X = RNG.normal(0, np.sqrt(1 / n), (n, p))
    Theta = X @ B
    # generate class label latent variables from Cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, n), np.cumsum(alpha))
    # generate Y by picking elements from Theta according to c
    Y = np.take_along_axis(Theta, c[:, None], axis=1)
    u = RNG.uniform(0, 1, (n, 1))
    Y = np.array(sigmoid(Y) > u, dtype=int)
    return X, Y, B, B_hat_0


def generate_data_mixed_relu(p, L, n, alpha, B_row_cov, sigma_sq, RNG=default_rng()):
    """
    Generate a random dataset according to the mixed rectified linear regression model.
    Parameters:
        p: int = number of dimensions
        L: int = number of mixture components
        n: int = number of samples
        alpha: L x 1 = categorical distribution on components
        B_row_cov: L x L = covariance matrix of the rows of B
        sigma_sq: float = noise variance
        RNG: numpy random number generator
    Returns:
        X: n x p = samples
        Y: n x 1 = observations
        B: p x L = signal matrix
        B_hat_0: p x L = initialisation of signal matrix estimate
    """
    # generate rows of signal matrix B according to zero mean Gaussian with covariance B_row_cov
    B = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    B_hat_0 = RNG.multivariate_normal(np.zeros(L), B_row_cov, p)
    X = RNG.normal(0, np.sqrt(1 / n), (n, p))
    Theta = X @ B
    # generate class label latent variables from Cat(alpha)
    c = np.digitize(RNG.uniform(0, 1, n), np.cumsum(alpha))
    # generate Y by picking elements from Theta according to c
    Y = np.take_along_axis(Theta, c[:, None], axis=1)
    Y = np.clip(Y, 0, np.inf)
    Y = Y + RNG.normal(0, sqrt(sigma_sq), n)[:, None]
    return X, Y, B, B_hat_0


def generate_alpha(L, scale=0.2, uniform=False, RNG=default_rng()):
    """
    Randomly generate a categorical distribution on L components.
    Parameters:
        L: int = number of components
        scale: float = scale parameter, controls how close the components proportions are to each other
        uniform: bool = whether to generate a uniform distribution
        RNG: numpy random number generator
    Returns:
        alpha: L x 1 = categorical distribution on components
    """
    if uniform:
        alpha = np.ones(L) / L
    else:
        alpha = RNG.uniform(scale, 1, L)
        alpha = alpha / np.linalg.norm(alpha, 1)
    return alpha


def isPSD(A, tol=1e-8):
    E = np.linalg.eigvalsh(A)
    return np.all(E > -tol)


def generate_B_row_cov(L, sigma_beta_sq, dependence=0.5, noise=0.3, RNG=default_rng()):
    """
    Generate a covariance matrix for the rows of B.
    Parameters:
        L: int = number of components
        sigma_beta_sq: float = variance of the rows
        dependence: float = controls the degree of dependence between the rows
        noise: float = controls the degree of variation between elements on the diagonal
        RNG: numpy random number generator
    Returns:
        B_row_cov: L x L = covariance matrix of the rows of B
    """
    B_cov = RNG.uniform(-dependence, dependence, (L, L))
    B_cov = sigma_beta_sq * B_cov @ B_cov.T
    B_diag = np.ones(L) * sigma_beta_sq + RNG.uniform(0, noise, L) * sigma_beta_sq
    np.fill_diagonal(B_cov, B_diag)
    while not isPSD(B_cov):
        B_cov += np.eye(L) * (noise + 1e-3) * sigma_beta_sq
    return B_cov
