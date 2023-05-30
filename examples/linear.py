import context
from gamp.run import run_trial
from gamp.plotting import plot_corr, plot_mse

# Set parameters ==============================
model = 'linear'
p = 500  # dimension
n = 2000  # number of samples
sigma_sq = 0.1  # noise variance
sigma_beta_sq = 20  # signal variance
n_iters = 5  # number of amp iterations

beta, beta_hat_list, mu_k_list = run_trial(model, p, n, sigma_sq, sigma_beta_sq, n_iters)
plot_mse(beta, beta_hat_list, mu_k_list, sigma_beta_sq)
plot_corr(beta, beta_hat_list, mu_k_list, sigma_beta_sq)
