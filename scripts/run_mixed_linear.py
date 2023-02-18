from itertools import permutations

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt

import context
from gamp.mlr import run_MLR_trial
from gamp.losses import B_loss

RNG = default_rng(2)
p = 250  # number of dimensions
L = 2  # number of mixture components
n = 2500  # number of samples
alpha = RNG.uniform(0, 0.1, L)
alpha = alpha / np.linalg.norm(alpha, 1)  # mixing proportions
sigma = 0.1  # noise variance
sigma_beta_sq = 1  # signal variance
n_iters = 15  # number of AMP iterations
# B_row_cov = np.eye(L)  # covariance matrix for rows of B
B_row_cov = RNG.uniform(-1, 1, (L, L))
B_row_cov = B_row_cov @ B_row_cov.T
B_row_cov += np.eye(L) * sigma_beta_sq
print(f"Sigma_B = \n{B_row_cov}")
print(f"alpha = {alpha}")


B, B_hat_list, M_k_B_list = run_MLR_trial(p, L, n, alpha, B_row_cov, sigma, n_iters)

mse = np.array([B_loss(B, B_hat) for B_hat in B_hat_list])

se = np.zeros_like(mse)
se[0, :] = 2 * np.diag(B_row_cov)

# state evolution
for j, M_k in enumerate(M_k_B_list, 1):
    Q = B_row_cov @ M_k.T @ inv(M_k @ B_row_cov @ M_k.T + M_k.T)
    for i in range(L):
        se[j, i] = B_row_cov[i, i] - Q[i, :] @ M_k @ B_row_cov[i, :]

colors = ['red', 'green', 'blue']
for i in range(L):
    plt.plot(range(len(B_hat_list)), mse[:, i], color=colors[i], label=f"l={i}, $\\alpha_i$={alpha[i]:.2f}")
    plt.plot(range(len(B_hat_list)), se[:, i], color=colors[i], linestyle='dashed', label=f"state evolution, l={i}")

plt.ylabel("Signal mean squared error")
plt.ylim(0, 2.2 * np.max(B_row_cov))
plt.xlabel("Iteration No.")
plt.legend()
plt.show()

