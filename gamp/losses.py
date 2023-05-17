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


def B_loss(B_hat, B, permute=False, B_cov=None):
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
        if B_cov is not None:
            return (norm(B - B_hat, axis=0) ** 2 / p) / np.diag(B_cov)
        return (norm(B - B_hat, axis=0) ** 2 / p)

    mse = np.repeat(B_hat[:, None].T, L, axis=1)
    mse = norm(mse - B.T, axis=2) ** 2 / p
    mse_list = np.zeros(L)
    for j in range(L):
        i = np.argmin(mse[:, j])
        mse_list[j] = mse[i, j]
        mse[i, :] = np.inf
    if B_cov is not None:
        return mse_list / np.diag(B_cov)
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


def prediction_error(beta, beta_hat, n, RNG, scale=10):
    p = beta.size
    X = RNG.normal(0, sqrt(1 / n), (scale*n, p))
    # data
    u = RNG.uniform(0, 1, scale*n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)

    # predictions:
    y_pred_hat = np.array(sigmoid(X @ beta_hat) > 0.5, dtype=int)
    return (scale*n - np.sum(y_pred_hat == y)) / (scale*n)


def prediction_error_mixed(B, B_hat, n, RNG, scale=10, combined=False, alpha=None):
    if combined:
        p, L = B.shape
        X = RNG.normal(0, np.sqrt(1 / n), (scale*n, p))
        Theta = X @ B
        # generate class label latent variables from Cat(alpha)
        c = np.digitize(RNG.uniform(0, 1, scale*n), np.cumsum(alpha))

        # generate Y by picking elements from Theta according to c
        Y = np.take_along_axis(Theta, c[:, None], axis=1)
        # plt.hist(sigmoid(Y), bins=10)
        # plt.show()
        u = RNG.uniform(0, 1, (scale*n, 1))
        Y = np.array(sigmoid(Y) > u, dtype=int).flatten()

        Y_pred = np.array(np.sum(alpha * sigmoid(X @ B_hat), axis=1) > 0.5, dtype=int).flatten()
        return (scale*n - np.sum(Y_pred == Y)) / (scale*n)
    return np.array([prediction_error(B[:, j], B_hat[:, j], n, RNG, scale) for j in range(B.shape[1])])


def logistic_log_loss(beta, beta_hat, n, RNG, scale=10):
    p = beta.size
    X = RNG.normal(0, sqrt(1 / n), (scale*n, p))
    # data
    u = RNG.uniform(0, 1, scale*n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int).flatten()
    y[y == 0] = -1
    return (- 1 / (scale*n)) * np.sum(np.log(sigmoid(y[:, None] * X @ beta_hat)))


def logistic_log_loss_mixed(B, B_hat, n, RNG, scale=10):
    return np.array([logistic_log_loss(B[:, j], B_hat[:, j], n, RNG, scale) for j in range(B.shape[1])])
