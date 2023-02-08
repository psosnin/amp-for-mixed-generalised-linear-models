from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression

from gamp.logistic import run_logistic_trial, sigmoid
from gamp.losses import beta_loss
from gamp.state_evolution import state_evolution_mse

RNG = np.random.default_rng(1)


def logistic_test(beta):
    # Representative logistic regression example
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    u = RNG.uniform(0, 1, n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)
    log_regression = LogisticRegression(fit_intercept=False)
    log_regression.fit(X, y)
    return norm(log_regression.coef_ - beta) ** 2 / p


# Set parameters ==============================
p = 500  # dimension
n = 2000  # number of samples
delta = n / p
sigma_beta_sq = 2  # signal variance
n_iters = 3  # number of amp iterations
n_trials = 5  # number of trials to run

mse = np.zeros((n_trials, n_iters + 1))
mse_se = np.zeros((n_trials, n_iters + 1))
mse_se[:, 0] = sigma_beta_sq * 2

lr = 0

for i in range(n_trials):
    beta, beta_hat_list, mu_k_list = run_logistic_trial(p, n, sigma_beta_sq, n_iters)
    mse[i, :] = np.array([beta_loss(beta, beta_hat) for beta_hat in beta_hat_list])
    mse_se[i, 1:] = np.array([state_evolution_mse(mu, mu, sigma_beta_sq) for mu in mu_k_list])
    lr += logistic_test(beta)

plt.axhline(lr / n_trials, label='logistic regression', color='green')

mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)

plt.errorbar(range(n_iters + 1), mse_mean, yerr=2*mse_std, color='blue', ecolor='blue', alpha=0.7,
             elinewidth=2, capsize=5, label='approximate message passing $\pm 2 \sigma$')

plt.plot(np.mean(mse_se, axis=0), color='red', label='state evolution')

plt.ylim(0, 1.1*max(np.max(mse_se), np.max(mse)))
plt.ylabel('Mean squared error')
plt.xlabel('Iteration No.')
plt.legend()
plt.show()
