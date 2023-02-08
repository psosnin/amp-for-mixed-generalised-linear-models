from math import sqrt

import numpy as np
from numpy.random import default_rng

RNG = default_rng(1)


def GAMP(X, y, beta_hat_k, sigma_beta_sq, sigma_sq, n_iters, gk_expect):
    """
    Run generalised approximate message passing to estimate beta from X and y.
    We assume a zero mean gaussian prior on the signal beta.
    Parameters:
        X: n x p = samples
        y: n x 1 = observations
        beta_hat_k: p x 1 = initial estimate of beta
        sigma_beta_sq: float = signal variance
        sigma_sq: float = noise variance
        n_iters: int = max number of AMP iterations to perform
        gk_expect: function to compute E[Z | Zk, Ybar], depends on choice of GLM
    Returns:
        beta_hat_list: list of estimates of B for each AMP iteration
        mu_k_list: list of state evolution mean for each iteration
    """
    n, p = X.shape
    delta = n / p

    # initialisation
    r_hat_k_minus_1 = np.zeros(n)
    b_k = 1
    # assumes beta_hat_0 generated from the same distribution as beta:
    Sigma_k = sigma_beta_sq * np.eye(2) / delta

    # containers to store results
    beta_hat_list = [beta_hat_k]
    mu_k_list = []

    # begin AMP iterations
    for _ in range(n_iters):
        # step 1:
        theta_k = X @ beta_hat_k - b_k * r_hat_k_minus_1
        # step 2:
        r_hat_k = apply_gk(theta_k, y, Sigma_k, sigma_sq, gk_expect)
        if np.isnan(r_hat_k).any():
            print(f"nan in r_hat_k at iteration {_}, stopping.")
            break
        # state evolution:
        mu_k_plus_1 = sigma_k_plus_1_sq = r_hat_k.T @ r_hat_k / n
        q = mu_k_plus_1 * sigma_beta_sq / (mu_k_plus_1 ** 2 * sigma_beta_sq + sigma_k_plus_1_sq)
        # step 3:
        c_k = compute_ck(theta_k, r_hat_k, Sigma_k, mu_k_plus_1)
        # step 4:
        beta_k_plus_1 = X.T @ r_hat_k - c_k * beta_hat_k
        # step 5:
        beta_hat_k_plus_1 = q * beta_k_plus_1
        # step 6:
        b_k_plus_1 = q / delta
        # update_sigma
        Sigma_k_plus_1 = update_Sigmak(q, mu_k_plus_1, sigma_beta_sq, delta)

        # prepare next iteration:
        beta_hat_k = beta_hat_k_plus_1
        Sigma_k = Sigma_k_plus_1
        r_hat_k_minus_1 = r_hat_k
        b_k = b_k_plus_1
        beta_hat_list.append(beta_hat_k)
        mu_k_list.append(mu_k_plus_1)

    return beta_hat_list, mu_k_list


def apply_gk(theta, y, Sigma_k, sigma_sq, gk_expect):
    """ Apply Bayesian-Optimal g_k* to each entry in theta and y. """
    Var_Z_given_Z_k = Sigma_k[0, 0] - Sigma_k[0, 1] ** 2 / Sigma_k[1, 1]
    theta_and_y = np.vstack((theta, y))
    return np.apply_along_axis(compute_gk_1d, 0, theta_and_y, Var_Z_given_Z_k, Sigma_k, sigma_sq, gk_expect)


def compute_gk_1d(Z_k_and_Y, Var_Z_given_Z_k, Sigma_k, sigma_sq, gk_expect):
    """ Compute the optial gk given Z_k and Y. """
    E_Z_given_Z_k = Z_k_and_Y[0] * Sigma_k[0, 1] / Sigma_k[1, 1]
    E_Z_given_Z_k_Y = gk_expect(Z_k_and_Y, Sigma_k, sigma_sq)
    return (E_Z_given_Z_k_Y - E_Z_given_Z_k) / Var_Z_given_Z_k


def compute_ck(theta_k, r_hat_k, Sigma_k, mu_k_plus_1):
    """ Approximate c_k from empirical averages. """
    return (np.mean(theta_k * r_hat_k) - Sigma_k[0, 1] * mu_k_plus_1) / Sigma_k[1, 1]


def update_Sigmak(q, mu_k_plus_1, sigma_beta_sq, delta):
    """ Update the state evolution. """
    S_12 = q * mu_k_plus_1 * sigma_beta_sq
    return np.block([
        [sigma_beta_sq, S_12],
        [S_12, S_12]
    ]) / delta
