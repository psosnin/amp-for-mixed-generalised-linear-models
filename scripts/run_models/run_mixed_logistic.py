from random import randint

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from tqdm import tqdm

import context
from gamp.mixed_logistic_vect import run_mixed_logistic_trial
from gamp.losses import B_loss, prediction_error_mixed
from gamp.state_evolution import state_evolution_mse_mixed

seed = randint(0, 10000)
# seed = 7416
print("seed = ", seed)
RNG = default_rng(seed)

# Set parameters ===========================================
p = 500  # number of dimensions
L = 3  # number of mixture components
n = 3000  # number of samples
sigma_beta_sq = 20 * n / p  # signal variance
n_iters = 10   # number of AMP iterations
n_trials = 1  # number of amp trials to perform

alpha = RNG.uniform(0.3, 1, L)
alpha = alpha / np.linalg.norm(alpha, 1)  # unequal mixing
alpha = np.ones(L) / L  # equal mixing

B_cov = RNG.uniform(-0.5, 0.5, (L, L))
B_cov = B_cov @ B_cov.T * sigma_beta_sq
B_diag = (np.ones(L) + RNG.uniform(0, 0.1, L)) * sigma_beta_sq
np.fill_diagonal(B_cov, B_diag)  # unequal variance dependent case
# B_cov = np.diag(B_diag)  # equal variance independent case
# B_cov = np.eye(L) * sigma_beta_sq  # equal variance independent case

print(f"Sigma_B = \n{B_cov}")
print(f"alpha = \n{alpha}")

mse = np.zeros((n_trials, n_iters + 1, L))
mse_se = np.zeros_like(mse)
mse_se[:, 0, :] = 2 * np.diag(B_cov)
pred = np.zeros((n_trials, n_iters + 1, L))

for i in tqdm(range(n_trials)):
    B, B_hat_list, M_k_B_list = run_mixed_logistic_trial(p, L, n, alpha, B_cov, n_iters, RNG)
    mse[i, :len(B_hat_list), :] = np.array([B_loss(B, B_hat) for B_hat in B_hat_list])
    mse[i, len(B_hat_list):] = mse[i, len(B_hat_list) - 1]
    pred[i, :len(B_hat_list), :] = np.array([prediction_error_mixed(B, B_hat, n, RNG) for B_hat in B_hat_list])
    pred[i, len(B_hat_list):] = pred[i, len(B_hat_list) - 1]
    mse_se[i, 1:len(B_hat_list), :] = np.array([state_evolution_mse_mixed(M_k, B_cov) for M_k in M_k_B_list])
    mse_se[i, len(B_hat_list):] = mse_se[i, len(B_hat_list) - 1]

    # B_hat = B_hat_list[-1]
    # if not np.allclose(B_loss(B, B_hat), B_loss(B, B_hat, True)):
    #     print(f"Warning, permutation in B_hat on trial {i}")

mse_se_mean = np.mean(mse_se, axis=0)
mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)

f, axs = plt.subplots(L, 1, sharex=True, sharey=True, figsize=(6, 6))

colors = ['red', 'green', 'blue', 'orange', 'pink', 'black', 'purple']
for i in range(L):
    for j in range(n_trials):
        axs[i].plot(range(n_iters + 1), mse[j, :, i], color=colors[i], alpha=0.3)
    # axs[i].errorbar(range(n_iters + 1), mse_mean[:, i], yerr=2*mse_std[:, i], color=colors[i],
    #                 alpha=0.7, elinewidth=2, capsize=5, label="mean squared error $\pm 2 \sigma_{MSE}$")
    axs[i].plot(range(n_iters + 1), mse_se_mean[:, i], color=colors[i], marker='o', fillstyle='none',
                linestyle='dashed', label=f"state evolution")
    axs[i].set_title(f"Mixture {i+1}, $\\alpha$={alpha[i]:.3f}")
    axs[i].set_ylim(0, 2.5 * np.max(B_cov))
    axs[i].set_ylabel("Signal MSE")
    axs[i].legend()

f.suptitle(
    f"AMP for Mixed Logistic Regression\n$L={L}, p={p}, n={n}, \ \sigma_\\beta^2={sigma_beta_sq}$")
plt.xlabel("Iteration No.")
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))

pred_mean = np.mean(pred, axis=0)
true_pred = prediction_error_mixed(B, B, n, RNG)
for i in range(L):
    plt.plot(pred_mean[:, i], color=colors[i], label=f'estimated B, l={i}')
    plt.axhline(true_pred[i], label=f'true B, l={i}', color=colors[i], linestyle='dotted')

plt.ylabel('Prediction error')
plt.xlabel('Iteration No.')
plt.ylim(0, 1)
plt.legend()
plt.show()
