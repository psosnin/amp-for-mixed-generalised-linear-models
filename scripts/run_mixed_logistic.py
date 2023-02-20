from random import randint

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from tqdm import tqdm

import context
from gamp.mixed_logistic import run_mixed_logistic_trial
from gamp.losses import B_loss
from gamp.state_evolution import state_evolution_mse_mixed

seed = randint(0, 10000)
# seed = 4347
print("seed = ", seed)
RNG = default_rng(seed)

# Set parameters ===========================================
p = 200  # number of dimensions
L = 2  # number of mixture components
n = 4000  # number of samples
sigma_beta_sq = 100  # signal variance
n_iters = 4  # number of AMP iterations
n_trials = 1  # number of amp trials to perform

alpha = RNG.uniform(0, 1, L)
alpha = alpha / np.linalg.norm(alpha, 1)  # unequal mixing
# alpha = np.ones(L) / L  # equal mixing

B_cov = RNG.uniform(-1, 1, (L, L))
B_cov = B_cov @ B_cov.T * sigma_beta_sq
B_diag = np.ones(L) * sigma_beta_sq + RNG.uniform(0, 0.3, L)
np.fill_diagonal(B_cov, B_diag)  # unequal variance dependent case
# B_cov = np.diag(B_diag)  # equal variance independent case
# B_cov = np.eye(L) * sigma_beta_sq  # equal variance independent case

print(f"Sigma_B = \n{B_cov}")
print(f"alpha = \n{alpha}")

mse = np.zeros((n_trials, n_iters + 1, L))
mse_se = np.zeros_like(mse)
mse_se[:, 0, :] = 2 * np.diag(B_cov)

for i in tqdm(range(n_trials)):
    B, B_hat_list, M_k_B_list = run_mixed_logistic_trial(p, L, n, alpha, B_cov, n_iters, RNG)
    mse[i, :len(B_hat_list), :] = np.array([B_loss(B, B_hat) for B_hat in B_hat_list])
    mse[i, len(B_hat_list):] = mse[i, len(B_hat_list) - 1]
    mse_se[i, 1:len(B_hat_list), :] = np.array([state_evolution_mse_mixed(M_k, B_cov) for M_k in M_k_B_list])
    mse_se[i, len(B_hat_list):] = mse_se[i, len(B_hat_list) - 1]

B_hat = B_hat_list[-1]
print(B_loss(B, B_hat,))
print(B_loss(B, B_hat, True))

mse_se_mean = np.mean(mse_se, axis=0)
mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)

plt.figure(figsize=(6, 6))

colors = ['red', 'green', 'blue', 'orange', 'pink', 'black', 'purple']
for i in range(L):
    for j in range(n_trials):
        plt.plot(range(n_iters + 1), mse[j, :, i], color=colors[i], alpha=0.7)
    # plt.errorbar(range(n_iters + 1), mse_mean[:, i], yerr=2*mse_std[:, i], color=colors[i],
    #              alpha=0.7, elinewidth=2, capsize=5, label=f"mean squared error $\pm 2 \sigma_e$, l={i}, $\\alpha_i$={alpha[i]:.2f}")
    plt.plot(range(n_iters + 1), mse_se_mean[:, i], color=colors[i],
             linestyle='dashed', label=f"state evolution, l={i}")

plt.title(
    f"AMP for Mixed Logistic Regression\n$L={L}, p={p}, n={n}, \ \sigma_\\beta^2={sigma_beta_sq}$\n$\\alpha = {np.round(alpha, 2)}$")
plt.ylabel("Signal mean squared error")
plt.ylim(0, 2.5 * np.max(B_cov))
plt.xlabel("Iteration No.")
plt.legend()
plt.show()