import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from gamp.linear import run_linear_trial
from gamp.losses import beta_loss
from gamp.state_evolution import state_evolution_mse

# Set parameters ==============================
sigma_sq = 0.1  # noise variance
p = 100  # dimension
n = 1000  # number of samples
delta = n / p
sigma_beta_sq = 1  # signal variance
n_iters = 10  # number of amp iterations
n_trials = 15  # number of trials to run

mse = np.zeros((n_trials, n_iters + 1))
mse_se = np.zeros((n_trials, n_iters + 1))
mse_se[:, 0] = sigma_beta_sq * 2

for i in range(n_trials):
    beta, beta_hat_list, mu_k_list = run_linear_trial(p, n, sigma_sq, sigma_beta_sq, n_iters)
    mse[i, :] = np.array([beta_loss(beta, beta_hat) for beta_hat in beta_hat_list])
    mse_se[i, 1:] = np.array([state_evolution_mse(mu, mu, sigma_beta_sq) for mu in mu_k_list])

mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)

plt.errorbar(range(n_iters + 1), mse_mean, yerr=2*mse_std, color='blue', ecolor='blue',
             alpha=0.7, elinewidth=2, capsize=5, label='mean squared error $\pm 2 \sigma$')

se_mean = np.mean(mse_se, axis=0)
se_std = np.std(mse_se, axis=0)

plt.errorbar(range(n_iters + 1), se_mean, yerr=2*se_std, color='red', ecolor='red',
             alpha=0.7, elinewidth=2, capsize=5, label='state evolution prediction $\pm 2 \sigma$')

plt.ylim(0, 1.1*max(np.max(mse_se), np.max(mse)))
plt.ylabel('Mean squared error')
plt.xlabel('Iteration No.')
plt.legend()
plt.show()
