from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.linear_model import LogisticRegression

import context
from gamp.logistic import run_logistic_trial, sigmoid
from gamp.losses import beta_loss, prediction_error
from gamp.state_evolution import state_evolution_mse

RNG = np.random.default_rng(1)


def logistic_test(beta):
    # Representative logistic regression example
    X = RNG.normal(0, sqrt(1 / n), (n, p))
    u = RNG.uniform(0, 1, n)
    y = np.array(sigmoid(X @ beta) > u, dtype=int)
    log_regression = LogisticRegression(fit_intercept=False)
    log_regression.fit(X, y)
    # return norm(log_regression.coef_ - beta) ** 2 / p
    return prediction_error(beta, log_regression.coef_.flatten(), n, RNG)


# Set parameters ==============================
p = 500  # dimension
n = 5000  # number of samples
delta = n / p
sigma_beta_sq = 20  # signal variance
n_iters = 5  # number of amp iterations
n_trials = 1  # number of trials to run

mse = np.zeros((n_trials, n_iters + 1))
mse_se = np.zeros((n_trials, n_iters + 1))
mse_se[:, 0] = sigma_beta_sq * 2

pred = np.zeros((n_trials, n_iters + 1))
lr = 0

for i in range(n_trials):
    beta, beta_hat_list, mu_k_list = run_logistic_trial(p, n, sigma_beta_sq, n_iters)
    mse[i, :] = np.array([beta_loss(beta, beta_hat) for beta_hat in beta_hat_list])
    pred[i, :] = np.array([prediction_error(beta, beta_hat, n, RNG) for beta_hat in beta_hat_list])
    mse_se[i, 1:] = np.array([state_evolution_mse(mu, mu, sigma_beta_sq) for mu in mu_k_list])
    lr += logistic_test(beta)

pred_mean = np.mean(pred, axis=0)
plt.plot(pred_mean, label='amp', color='red')
plt.axhline(prediction_error(beta, beta, n, RNG), label='true parameters', color='blue')
plt.axhline(lr / n_trials, label='scikit logistic', color='green')
plt.ylabel('Prediction error')
plt.xlabel('Iteration No.')
plt.ylim(0, 1)
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))

mse_mean = np.mean(mse, axis=0)
mse_std = np.std(mse, axis=0)

plt.errorbar(range(n_iters + 1), mse_mean, yerr=2*mse_std, color='blue', ecolor='blue', alpha=0.6,
             elinewidth=2, capsize=5, label='amp mse $\pm 2 \sigma_{MSE}$')

plt.plot(np.mean(mse_se, axis=0), color='red', label='state evolution')

plt.ylim(0, 1.1*max(np.max(mse_se), np.max(mse)))

plt.title(f"AMP for Logistic Regression\n$p={p}, n={n}, \ \sigma_\\beta^2={sigma_beta_sq}$")
plt.ylabel('Signal mean squared error')
plt.xlabel('Iteration No.')
plt.legend()
plt.savefig('plots/logistic.png')
plt.show()
