import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_sklearn(Y, X, w):
    log_regression = LogisticRegression(fit_intercept=False)
    log_regression.fit(X, Y.flatten(), w)
    return log_regression.coef_.flatten()


def EM_mixed_logistic(X, Y, B_hat_k, alpha, n_iters):
    """
    Run expectation maximisation to estimate B from X and Y in the
    mixed logistic model.
    Parameters:
        X: n x p = samples
        Y: n x 1 = observations
        B_hat_k: p x L = initial estimate of signal matrix
        alpha: L x 1 = categorical distribution on components
        grad: bool = if true use gradient ascent otherwise use sklearn log regression
        n_iters: int = max number of EM iterations
    Returns:
        B_hat_list: list of estimates of B for each EM
    """
    n, p = X.shape
    L = alpha.size

    # transform Y from {0, 1} to {-1, 1}
    Y[Y == 0] = -1
    B_hat_list = [B_hat_k]
    for k in range(n_iters):
        # E-step: compute posterior given B
        W = alpha * sigmoid(Y * X @ B_hat_k)
        W = W / norm(W, 1, axis=1)[:, None]

        # M-step: estimate B given posterior
        B_hat_k_plus_1 = np.zeros_like(B_hat_k)
        for i in range(L):
            B_hat_k_plus_1[:, i] = fit_sklearn(Y, X, W[:, i])
        B_hat_list.append(B_hat_k_plus_1)
        B_hat_k = B_hat_k_plus_1
    return B_hat_list
