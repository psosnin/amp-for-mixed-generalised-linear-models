from random import randint

import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from tqdm import tqdm

import context
from gamp.mixed_logistic_vect import run_mixed_logistic_trial
from em.logistic_em import EM_mixed_logistic
from gamp.losses import prediction_error_mixed

seed = randint(0, 10000)
print("seed = ", seed)
RNG = default_rng(seed)

# Set parameters ===========================================
p = 500  # number of dimensions
L = 2  # number of mixture components
n = 5000  # number of samples
sigma_beta_sq = 5 * n / p  # signal variance
n_iters = 10   # number of AMP iterations
n_trials = 5  # number of amp trials to perform

alpha = RNG.uniform(0.3, 1, L)
alpha = alpha / np.linalg.norm(alpha, 1)  # unequal mixing

B_cov = RNG.uniform(-0.5, 0.5, (L, L))
B_cov = B_cov @ B_cov.T * sigma_beta_sq
B_diag = (np.ones(L) + RNG.uniform(0, 0.1, L)) * sigma_beta_sq
np.fill_diagonal(B_cov, B_diag)  # unequal variance dependent case
# B_cov = np.diag(B_diag)  # equal variance independent case
# B_cov = np.eye(L) * sigma_beta_sq  # equal variance independent case

print(f"Sigma_B = \n{B_cov}")
print(f"alpha = \n{alpha}")

B, B_hat_list, M_k_B_list, X, Y, B_hat_0 = run_mixed_logistic_trial(p, L, n, alpha, B_cov, n_iters, RNG, True)
amp_losses = np.array([prediction_error_mixed(B, B_hat, n, RNG) for B_hat in B_hat_list])

B_hat_list_EM = EM_mixed_logistic(X, Y, B_hat_0, alpha, n_iters)
em_losses = np.array([prediction_error_mixed(B, B_hat, n, RNG) for B_hat in B_hat_list_EM])

best_prediction_error = prediction_error_mixed(B, B, n, RNG)

f, axs = plt.subplots(L, 1, sharex=True, sharey=True, figsize=(6, 6))

colors = ['red', 'green', 'blue', 'orange', 'pink', 'black', 'purple']
for i in range(L):
    axs[i].axhline(best_prediction_error[i], color=colors[i], label="True parameters")
    axs[i].plot(amp_losses[:, i], color=colors[i], label=f"AMP")
    axs[i].plot(em_losses[:, i], color=colors[i], linestyle='dashed', label="EM")
    axs[i].set_title(f"Mixture {i+1}, $\\alpha$={alpha[i]:.3f}")
    axs[i].set_ylabel("Prediction error")
    axs[i].legend()

f.suptitle(
    f"AMP vs EM for Mixed Logistic Regression\n$L={L}, p={p}, n={n}, \ \sigma_\\beta^2={sigma_beta_sq}$")
plt.xlabel("Iteration No.")
plt.show()
