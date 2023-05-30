import numpy as np

import context
from gamp.run import run_trial_mixed
from gamp.generate_data import generate_alpha, generate_B_row_cov
from gamp.plotting import plot_mse_mixed, plot_corr_mixed

# Set parameters ==============================
RNG = np.random.default_rng()
model = 'relu'
p = 2500  # dimension
L = 2  # number of mixtures
n = 10000  # number of samples
alpha = generate_alpha(L, 0.5, False, RNG)  # component proportions
sigma_beta_sq = 1  # signal variance
B_row_cov = generate_B_row_cov(L, sigma_beta_sq, 1, 0.5, RNG)  # component row covariances
sigma_sq = 0.001  # noise variance
n_iters = 15  # number of amp iterations

print('alpha: \n', alpha)
print('B_row_cov: \n', B_row_cov)

B, B_hat_list, M_k_list = run_trial_mixed(model, p, L, n, alpha, B_row_cov, sigma_sq, n_iters, RNG)
plot_mse_mixed(B, B_hat_list, M_k_list, B_row_cov, alpha)
plot_corr_mixed(B, B_hat_list, M_k_list, B_row_cov, alpha)
