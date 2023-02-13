import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import context
from gamp.logistic import run_logistic_trial
from gamp.losses import norm_sq_corr
from gamp.state_evolution import state_evolution_corr

# Set parameters ==============================
p = 200  # dimension
sigma_beta_sq = 4  # signal variance
n_iters = 3  # number of amp iterations
n_trials = 10  # number of trials to run
n_deltas = 10  # number of deltas to try

delta_list = np.linspace(0.1, 3, n_deltas)
n_list = np.array(p * delta_list, dtype=int)

mse = np.zeros((n_deltas, n_trials))
mse_se = np.zeros((n_deltas, n_trials))
for j, n in enumerate(tqdm(n_list)):
    for i in range(n_trials):
        beta, beta_hat_list, mu_k_list = run_logistic_trial(p, n, sigma_beta_sq, n_iters)
        mse[j, i] = norm_sq_corr(beta, beta_hat_list[-1])
        mse_se[j, i] = state_evolution_corr(mu_k_list[-1], mu_k_list[-1], sigma_beta_sq)

mse_mean = np.mean(mse, axis=1)
mse_std = np.std(mse, axis=1)

plt.figure(figsize=(6, 6))

plt.errorbar(delta_list, mse_mean, yerr=2*mse_std, color='blue', ecolor='blue',
             alpha=0.7, elinewidth=2, capsize=5, label='mean squared error $\pm 2 \sigma_{MSE}$')

plt.plot(delta_list, np.mean(mse_se, axis=1), color='red', marker='o', fillstyle='none', label='state evolution')

plt.ylim(0, 1.1*max(np.max(mse_se), np.max(mse)))
plt.title(f"AMP for Logistic Regression\n$p={p},\ \sigma_\\beta^2={sigma_beta_sq}$")
plt.ylabel('Signal normalised square correlation')
plt.xlabel('delta = n / p')
plt.legend()
plt.savefig("plots/logistic_vary_delta_corr.png")
plt.show()
