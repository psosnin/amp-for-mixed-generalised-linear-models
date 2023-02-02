import numpy as np


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


def beta_loss(B_hat, B):
    """
        Find the sum of the mean squared error between the true regression slopes and their estimates.
    """
    J = B_hat.shape[1]
    mse = np.repeat(B_hat[:, None].T, J, axis=1)
    mse = np.mean(np.square(mse - B.T), axis=2)
    total = 0
    while mse.size:
        i, j = np.unravel_index(mse.argmin(), mse.shape)
        total += mse[i, j]
        mse = np.delete(mse, i, 0)
        mse = np.delete(mse, j, 1)
    return total / J


def state_evolution_mse(M_k_B):
    """ Mean squared error according to the state evolution"""
    return np.trace(np.linalg.pinv(M_k_B)) / M_k_B.shape[0]
