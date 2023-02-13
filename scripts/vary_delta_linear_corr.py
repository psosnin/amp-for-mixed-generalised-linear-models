import numpy as np
import matplotlib.pyplot as plt

import context
from gamp.linear import run_linear_trial
from gamp.losses import norm_sq_corr
from gamp.state_evolution import state_evolution_corr

# Set parameters ==============================
sigma_sq = 0.1  # noise variance
p = 500  # dimension
sigma_beta_sq = 1  # signal variance
n_iters = 5  # number of amp iterations
n_trials = 10  # number of trials to run
n_deltas = 15  # number of deltas to try

delta_list = np.linspace(0.01, 2, n_deltas)
n_list = np.array(p * delta_list, dtype=int)

corr = np.zeros((n_deltas, n_trials))
corr_se = np.zeros((n_deltas, n_trials))
for j, n in enumerate(n_list):
    for i in range(n_trials):
        beta, beta_hat_list, mu_k_list = run_linear_trial(p, n, sigma_sq, sigma_beta_sq, n_iters)
        corr[j, i] = norm_sq_corr(beta, beta_hat_list[-1])
        corr_se[j, i] = state_evolution_corr(mu_k_list[-1], mu_k_list[-1], sigma_beta_sq)

mse_mean = np.mean(corr, axis=1)
mse_std = np.std(corr, axis=1)

plt.figure(figsize=(6, 6))

plt.errorbar(delta_list, mse_mean, yerr=2*mse_std, color='blue', ecolor='blue',
             alpha=0.7, elinewidth=2, capsize=5, label='mean squared error $\pm 2 \sigma_{MSE}$')

plt.plot(delta_list, np.mean(corr_se, axis=1), color='red', marker='o', fillstyle='none', label='state evolution')

plt.ylim(0, 1.1*max(np.max(corr_se), np.max(corr)))
plt.title(f"AMP for Linear Regression\n$p={p},\ \sigma^2={sigma_sq}, \ \sigma_\\beta^2={sigma_beta_sq}$")
plt.ylabel('Signal normalised squared correlation')
plt.xlabel('delta = n / p')
plt.legend()
plt.savefig("plots/linear_vary_delta_corr.png")
plt.show()
