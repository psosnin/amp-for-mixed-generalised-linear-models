import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import context
from gamp.mlr import run_MLR_trial
from gamp.losses import B_loss

RNG = default_rng(1)
p = 500  # number of dimensions
L = 2  # number of mixture components
n = 2500  # number of samples
alpha = RNG.uniform(0, 0.1, L)
alpha = alpha / np.linalg.norm(alpha, 1)  # mixing proportions
sigma = 0.1  # noise variance
n_iters = 15  # number of AMP iterations
B, B_hat_list, M_k_B_list = run_MLR_trial(p, L, n, alpha, sigma, n_iters)
losses = [B_loss(B, B_hat) for B_hat in B_hat_list]
x = np.arange(0, len(B_hat_list))
plt.plot(x, losses)
plt.ylim(0, np.max(losses)*1.1)
plt.xlabel('Iteration No.')
plt.ylabel('Mean Squared Error')
plt.show()
