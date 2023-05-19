from .generate_data import *
from .fitting.gamp import GAMP
from .fitting.matrix_gamp import matrix_GAMP
from .gk import linear, logistic, relu
from .gk import mixed_linear_vect as mixed_linear
from .gk import mixed_logistic_vect as mixed_logistic
from .gk import mixed_relu_vect as mixed_relu


"""
Functions for running a single trial of GAMP or EM/AM.
"""


def run_trial(model, algorithm, p, n, sigma_sq, sigma_beta_sq, n_iters, RNG=None):
    """
    Run a single trial with the specified (non-mixed) glm and algorithm.
    Parameters:
        model: str = 'linear', 'logistic' or 'relu'
        algorithm: str = 'GAMP', 'EM', 'AM'
        p: int = number of dimensions
        n: int = number of samples
        sigma_sq: float = noise variance
        sigma_beta_sq: float = signal variance
        n_iters: int = max number of iterations to perform
        RNG: = np random number generator
    Returns:
        beta: np.array = true signal
        beta_hat_list: list of np.array = list of estimated signals
        mu_k_list: list of np.array = list of state evolution means
    """
    if RNG is None:
        RNG = np.random.default_rng()
    if model == 'linear' and algorithm == 'GAMP':
        X, y, beta, beta_hat_0 = generate_data_linear(p, n, sigma_sq, sigma_beta_sq, RNG)
        beta_hat_list, mu_k_list = GAMP(X, y, beta_hat_0, sigma_beta_sq, sigma_sq, n_iters, linear.apply_gk)
    elif model == 'logistic' and algorithm == 'GAMP':
        X, y, beta, beta_hat_0 = generate_data_logistic(p, n, 0, sigma_beta_sq, RNG)
        beta_hat_list, mu_k_list = GAMP(X, y, beta_hat_0, sigma_beta_sq, 0, n_iters, logistic.apply_gk)
    elif model == 'relu' and algorithm == 'GAMP':
        X, y, beta, beta_hat_0 = generate_data_relu(p, n, sigma_sq, sigma_beta_sq, RNG)
        beta_hat_list, mu_k_list = GAMP(X, y, beta_hat_0, sigma_beta_sq, sigma_sq, n_iters, relu.apply_gk)
    else:
        raise NotImplementedError
    return beta, beta_hat_list, mu_k_list


def run_trial_mixed(model, algorithm, p, L, n, alpha, B_row_cov, sigma_sq, n_iters, spectral=False, RNG=None):
    """
    Run a single trial with the specified mixed glm and algorithm.
    Parameters:
        model: str = 'linear', 'logistic' or 'relu'
        algorithm: str = 'GAMP', 'EM', 'AM'
        p: int = number of dimensions
        L: int = number of mixtures
        n: int = number of samples
        alpha: float = signal variance
        B_row_cov: np.array = covariance matrix of the rows of B
        sigma_sq: float = noise variance
        n_iters: int = max number of iterations to perform
        spectral: bool = whether to use spectral initialization
        RNG: = np random number generator
    Returns:
        B: np.array = true signal
        B_hat_list: list of np.array = list of estimated signals
        M_k_list: list of np.array = list of state evolution mu_k
    """
    if RNG is None:
        RNG = np.random.default_rng()
    if model == 'linear' and algorithm == 'GAMP':
        X, Y, B, B_hat_0 = generate_data_mixed_linear(p, L, n, alpha, B_row_cov, sigma_sq, spectral, RNG)
        B_hat_list, M_k_list = matrix_GAMP(X, Y, B_hat_0, B_row_cov, sigma_sq, alpha, n_iters, mixed_linear.apply_gk)
    elif model == 'logistic' and algorithm == 'GAMP':
        X, Y, B, B_hat_0 = generate_data_mixed_logistic(p, L, n, alpha, B_row_cov, sigma_sq, RNG)
        B_hat_list, M_k_list = matrix_GAMP(X, Y, B_hat_0, B_row_cov, sigma_sq, alpha, n_iters, mixed_logistic.apply_gk)
    elif model == 'relu' and algorithm == 'GAMP':
        X, Y, B, B_hat_0 = generate_data_mixed_relu(p, L, n, alpha, B_row_cov, sigma_sq, RNG)
        B_hat_list, M_k_list = matrix_GAMP(X, Y, B_hat_0, B_row_cov, sigma_sq, alpha, n_iters, mixed_relu.apply_gk)
    else:
        raise NotImplementedError
    return B, B_hat_list, M_k_list
