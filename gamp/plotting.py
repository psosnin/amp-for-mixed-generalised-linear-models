import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .losses import mse, state_evolution_mse
from .losses import norm_sq_corr, state_evolution_corr
from .losses import mse_mixed, state_evolution_mse_mixed

sns.set_theme(context='paper', palette='Dark2', style='ticks')


def plot_mse(beta, beta_hat_list, mu_k_list, sigma_beta_sq, filename=None):
    """
    Plot the mean squared error of the signal estimate vs. the number of iterations.
    Parameters:
        beta: true signal
        beta_list: list of signal estimates
        mu_k_list: list of state evolution means
        sigma_beta_sq: signal variance
        filename: filename to save the plot
    """
    losses = np.array([mse(beta, beta_hat, sigma_beta_sq) for beta_hat in beta_hat_list])
    losses_se = np.array([2] + [state_evolution_mse(mu, mu, sigma_beta_sq, True) for mu in mu_k_list])
    plt.figure(figsize=(5, 5))
    plt.plot(range(len(losses)), losses, label='amp')
    plt.plot(range(len(losses_se)), losses_se, label='state evolution prediction')
    plt.ylim(0, 2.3)
    plt.xlabel('Iteration No.')
    plt.ylabel('Normalised mean squared error')
    plt.legend()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_corr(beta, beta_hat_list, mu_k_list, sigma_beta_sq, filename=None):
    """
    Plot the mean squared error of the signal estimate vs. the number of iterations.
    Parameters:
        beta: true signal
        beta_list: list of signal estimates
        mu_k_list: list of state evolution means
        sigma_beta_sq: signal variance
        filename: filename to save the plot
    """
    losses = np.array([norm_sq_corr(beta, beta_hat) for beta_hat in beta_hat_list])
    losses_se = np.array([0] + [state_evolution_corr(mu, mu, sigma_beta_sq) for mu in mu_k_list])
    plt.figure(figsize=(5, 5))
    plt.plot(range(len(losses)), losses, label='amp')
    plt.plot(range(len(losses_se)), losses_se, label='state evolution prediction')
    plt.xlabel('Iteration No.')
    plt.ylabel('Normalised squared correlation')
    plt.ylim(0, 1)
    plt.legend()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def plot_mse_mixed(B, B_hat_list, M_k_list, B_cov, alpha, spectral=False, permute=False, filename=None):
    """
    Plot the mean squared error of the signal estimate vs. the number of iterations.
    Parameters:
        B: true signal matrix
        B_hat_list: list of signal estimates
        M_k_list: list of state evolution means
        B_cov: signal covariance matrix
        alpha: mixture proportions
        spectral: whether spectral initialisation was used
        filename: filename to save the plot
    """
    L = B.shape[1]
    q = 1 if spectral else 2
    palette = sns.color_palette('Dark2', L)
    losses = np.array([mse_mixed(B, B_hat, B_cov, permute) for B_hat in B_hat_list])
    losses_se = np.array([[q] * L] + [state_evolution_mse_mixed(M_k, B_cov, True) for M_k in M_k_list])
    f, axs = plt.subplots(L, 1, sharex=True, sharey=True, figsize=(5, 5))
    for i in range(L):
        axs[i].axhline(1, color=palette[i], linestyle='dotted')
        axs[i].plot(range(len(losses)), losses[:, i], color=palette[i], label='amp')
        axs[i].plot(range(len(losses_se)), losses_se[:, i], linestyle='dashed',
                    color=palette[i], label='state evolution prediction')
        axs[i].set_ylabel('Normalised mean squared error')
        axs[i].set_title(f"Mixture {i+1}, $\\alpha$={alpha[i]:.3f}, $\\sigma_\\beta^2$={B_cov[i, i]:.3f}")
        axs[i].set_ylim(0, 1.2 * q)
        axs[i].legend()
    plt.xlabel('Iteration No.')
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()
