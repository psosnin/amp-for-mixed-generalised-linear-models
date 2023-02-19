from itertools import permutations

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from tqdm import tqdm

import context
from gamp.mlr import run_MLR_trial
from gamp.losses import B_loss
from gamp.state_evolution import state_evolution_mse_mixed

# Set parameters ===========================================
RNG = default_rng()
p = 250  # number of dimensions
L = 3  # number of mixture components
n = 1000  # number of samples
alpha = RNG.uniform(0, 0.1, L)
alpha = alpha / np.linalg.norm(alpha, 1)  # mixing proportions
sigma_sq = 0.1  # noise variance
sigma_beta_sq = 1  # signal variance
n_iters = 5  # number of AMP iterations
n_trials = 10  # number of amp trials to perform
B_cov = RNG.uniform(-0.5, 0.5, (L, L))
B_cov = B_cov @ B_cov.T
B_diag = np.ones(L) * sigma_beta_sq + RNG.uniform(0, 0.5, L)
print(B_diag)
np.fill_diagonal(B_cov, B_diag)
print(f"Sigma_B = \n{B_cov}")
print(f"alpha = {alpha}")

mse = np.zeros((n_trials, n_iters + 1, L))
mse_se = np.zeros_like(mse)
mse_se[:, 0, :] = 2 * np.diag(B_cov)

for i in tqdm(range(n_trials)):
    B, B_hat_list, M_k_B_list = run_MLR_trial(p, L, n, alpha, B_cov, sigma_sq, n_iters)
    mse[i, :len(B_hat_list), :] = np.array([B_loss(B, B_hat) for B_hat in B_hat_list])
    mse[i, len(B_hat_list):] = mse[i, len(B_hat_list) - 1]
    mse_se[i, 1:len(B_hat_list), :] = np.array([state_evolution_mse_mixed(M_k, B_cov) for M_k in M_k_B_list])
    mse_se[i, len(B_hat_list):] = mse_se[i, len(B_hat_list) - 1]

mse_se_mean = np.mean(mse_se, axis=0)
mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)

plt.figure(figsize=(6, 6))

colors = ['red', 'green', 'blue']
for i in range(L):
    plt.errorbar(range(n_iters + 1), mse_mean[:, i], yerr=2*mse_std[:, i], color=colors[i],
                 alpha=0.7, elinewidth=2, capsize=5, label='mean squared error $\pm 2 \sigma_{MSE}$, l={i}, $\\alpha_i$={alpha[i]:.2f}')
    plt.plot(range(n_iters + 1), mse_se_mean[:, i], color=colors[i], linestyle='dashed', label=f"state evolution, l={i}")

plt.title(f"AMP for Mixed Linear Regression\n$L={L}, p={p}, n={n}, \ \sigma^2={sigma_sq}, \ \sigma_\\beta^2={sigma_beta_sq}$\n$\\alpha = {np.round(alpha, 2)}$")
plt.ylabel("Signal mean squared error")
plt.ylim(0, 2.2 * np.max(B_cov))
plt.xlabel("Iteration No.")
plt.legend()
plt.show()

