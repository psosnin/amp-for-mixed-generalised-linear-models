import numpy as np
import matplotlib.pyplot as plt

import context
from gamp.linear import run_linear_trial
from gamp.losses import beta_loss
from gamp.state_evolution import state_evolution_mse

# Set parameters ==============================
sigma_sq = 0.1  # noise variance
p = 500  # dimension
sigma_beta_sq = 1  # signal variance
n_iters = 5  # number of amp iterations
n_trials = 10  # number of trials to run
n_deltas = 10  # number of deltas to try

delta_list = np.linspace(0.5, 4, n_deltas)
n_list = np.array(p * delta_list, dtype=int)

mse = np.zeros((n_deltas, n_trials))
mse_se = np.zeros((n_deltas, n_trials))
for j, n in enumerate(n_list):
    for i in range(n_trials):
        beta, beta_hat_list, mu_k_list = run_linear_trial(p, n, sigma_sq, sigma_beta_sq, n_iters)
        mse[j, i] = beta_loss(beta, beta_hat_list[-1])
        mse_se[j, i] = state_evolution_mse(mu_k_list[-1], mu_k_list[-1], sigma_beta_sq)

mse_mean = np.mean(mse, axis=1)
mse_std = np.std(mse, axis=1)

plt.figure(figsize=(6, 6))

plt.errorbar(delta_list, mse_mean, yerr=2*mse_std, color='blue', ecolor='blue',
             alpha=0.7, elinewidth=2, capsize=5, label='mean squared error $\pm 2 \sigma_{MSE}$')

plt.plot(delta_list, np.mean(mse_se, axis=1), color='red', marker='o', fillstyle='none', label='state evolution')

plt.ylim(0, 1.1*max(np.max(mse_se), np.max(mse)))

plt.title(f"AMP for Linear Regression\n$p={p},\ \sigma^2={sigma_sq}, \ \sigma_\\beta^2={sigma_beta_sq}$")
plt.ylabel('Signal mean squared error')
plt.xlabel('delta = n / p')
plt.legend()
plt.savefig("plots/linear_vary_delta.png")
plt.show()
