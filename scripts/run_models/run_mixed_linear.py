from itertools import permutations
from random import randint

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from tqdm import tqdm

import context
from gamp.mixed_linear_vect import run_MLR_trial
from gamp.losses import B_loss
from gamp.state_evolution import state_evolution_mse_mixed
from em.linear_em import EM

seed = randint(0, 10000)
seed = 1841
print("seed = ", seed)
RNG = default_rng(seed)

# Set parameters ===========================================
p = 500  # number of dimensions
L = 3  # number of mixture components
n = 2500  # number of samples
sigma_sq = 0.001  # noise variance
sigma_beta_sq = 1  # signal variance
n_iters = 10  # number of AMP iterations
n_trials = 100  # number of amp trials to perform

# alpha = RNG.uniform(0.2, 1, L)
# alpha = alpha / np.linalg.norm(alpha, 1)  # unequal mixing
# alpha = np.ones(L) / L  # equal mixing
alpha = np.array([0.5, 0.2, 0.3])

B_cov = RNG.uniform(-0.5, 0.5, (L, L))
B_cov = B_cov @ B_cov.T
B_diag = np.ones(L) * sigma_beta_sq + RNG.uniform(0, 0.3, L)
np.fill_diagonal(B_cov, B_diag)  # unequal variance dependent case
# B_cov = np.diag(B_diag)  # equal variance independent case
# B_cov = np.eye(L) * sigma_beta_sq  # equal variance independent case

print(f"Sigma_B = \n{B_cov}")
print(f"alpha = \n{alpha}")

mse = np.zeros((n_trials, n_iters + 1, L))
mse_EM = np.zeros((n_trials, n_iters + 1, L))
mse_se = np.zeros_like(mse)
mse_se[:, 0, :] = 2 * np.diag(B_cov)

for i in tqdm(range(n_trials)):
    B, B_hat_list, M_k_B_list, X, Y, B_hat_0 = run_MLR_trial(
        p, L, n, alpha, B_cov, sigma_sq, n_iters, RNG, True)
    # EM_B_list = EM(X, Y, B_hat_0, alpha, sigma_sq, n_iters)
    # mse_EM[i, :, :] = np.array([B_loss(B, B_hat, True) for B_hat in EM_B_list])
    mse[i, :len(B_hat_list), :] = np.array(
        [B_loss(B, B_hat) for B_hat in B_hat_list])
    mse[i, len(B_hat_list):] = mse[i, len(B_hat_list) - 1]
    mse_se[i, 1:len(B_hat_list), :] = np.array(
        [state_evolution_mse_mixed(M_k, B_cov) for M_k in M_k_B_list])
    mse_se[i, len(B_hat_list):] = mse_se[i, len(B_hat_list) - 1]

B_hat = B_hat_list[-1]
print(B_loss(B, B_hat,))
print(B_loss(B, B_hat, True))

mse_se_mean = np.mean(mse_se, axis=0)
mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)
mse_EM_mean = np.mean(mse_EM, axis=0)
mse_EM_std = np.std(mse_EM, axis=0)

f, axs = plt.subplots(L, 1, sharex=True, sharey=True, figsize=(8, 8))

colors = ['red', 'green', 'blue', 'orange', 'pink', 'black', 'purple']
for i in range(L):
    # for j in range(n_trials):
    #     axs[i].plot(range(n_iters + 1), mse[j, :, i],
    #                 color=colors[i], alpha=0.3)
    # axs[i].plot(range(n_iters + 1), mse_EM[j, :, i],
    #             linestyle='dotted', color=colors[i], alpha=0.3)
    axs[i].errorbar(range(n_iters + 1), mse_mean[:, i], yerr=2*mse_std[:, i], color=colors[i],
                    alpha=0.7, elinewidth=2, capsize=5, label=f"mean squared error $\pm 2 \sigma_e$, l={i}, $\\alpha_i$={alpha[i]:.2f}")
    axs[i].plot(range(n_iters + 1), mse_se_mean[:, i], color=colors[i], marker='o',
                linestyle='dashed', label=f"state evolution")
    axs[i].set_title(
        f"Mixture {i+1}, $\\alpha$={alpha[i]:.3f}, $\\sigma_\\beta^2$={B_cov[i, i]:.3f}")
    axs[i].set_ylim(0, 2.5 * np.max(B_cov))
    axs[i].set_ylabel("Signal MSE")
    axs[i].legend()

f.suptitle(
    f"AMP for Mixed Linear Regression\n"
    + f"$L={L}, p={p}, n={n}, \ \sigma^2={sigma_sq}, \ \sigma_\\beta^2={sigma_beta_sq}$")

plt.xlabel("Iteration No.")
plt.savefig("plots/mixed_linear.png", bbox_inches='tight')
plt.show()
