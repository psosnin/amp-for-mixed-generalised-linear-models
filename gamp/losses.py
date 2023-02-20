from math import sqrt

import numpy as np
from numpy.linalg import norm
from .logistic import sigmoid


def MSE(X, y, B_hat):
    """
        Returns the MSE of the predictions of the model vs the true output.
        Parameters:
            X: N x k+1 matrix, rows are xi samples
            y: N x 1 matrix, elements are mlr output
            B_hat: k+1 x J matrix, estimate of regression slopes
        Returns:
            mse: mean squared error of the predictions of the model
    """
    y_hat = X@B_hat
    return np.mean(np.min(np.square(y_hat - y), axis=1))


def B_loss(B_hat, B, permute=False):
    """
        Find the sum of the mean squared error between the true regression slopes and their estimates.
        Note that B_hat may have permuted columns compared to B.
        Parameters:
            B_hat: p x L = estimate of signal matrix
            B: p x L = true signal matrix
            permute: bool = whether to consider permutations of B_hat matrix
        Returns:
            mse: mean squared error of estimated signal and true signal
    """
    p, L = B_hat.shape
    if not permute:
        return norm(B - B_hat, axis=0) ** 2 / p

    mse = np.repeat(B_hat[:, None].T, L, axis=1)
    mse = norm(mse - B.T, axis=2) ** 2 / p
    mse_list = np.zeros(L)
    for j in range(L):
        i = np.argmin(mse[:, j])
        mse_list[j] = mse[i, j]
        mse[i, :] = np.inf
    return mse_list


def beta_loss(beta, beta_hat):
    """
    MSE between signal and estimate normalised by the dimension
    """
    return norm(beta - beta_hat) ** 2 / beta.shape[0]


def norm_sq_corr(beta, beta_hat):
    """
    Calculate the normalised squared correlation between beta and beta_hat.
    """
    return (np.dot(beta, beta_hat) / (norm(beta) * norm(beta_hat))) ** 2

def prediction_error(beta, beta_hat, n, RNG):
    p = beta.size
    X = RNG.normal(0, sqrt(1 / n), (5*n, p))
    # data
    u = RNG.uniform(0, 1, 5*n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)

    # predictions:
    y_pred_hat = np.array(sigmoid(X @ beta_hat) > 0.5, dtype=int)
    return (5*n - np.sum(y_pred_hat == y)) / (5*n)

def prediction_error_mixed(B, B_hat, n, RNG):
    return np.array([prediction_error(B[:, j], B_hat[:, j], n, RNG) for j in range(B.shape[1])])