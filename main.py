import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from gamp.mlr import run_MLR_trial
from gamp.losses import beta_loss

RNG = default_rng(1)

p = 25  # number of dimensions
L = 2  # number of components
n = 2500  # number of samples
alpha = RNG.uniform(0, 0.1, L)
alpha = alpha / np.linalg.norm(alpha, 1)
print(alpha)
sigma = 0.1
n_iters = 15  # number of AMP iterations

B, B_hat_list, M_k_B_list = run_MLR_trial(p, L, n, alpha, sigma, n_iters)
true_losses = [beta_loss(B, B_hat) for B_hat in B_hat_list]
x = np.arange(0, len(B_hat_list))
print(true_losses[-1])
plt.plot(x, true_losses, label='signal mse')
plt.ylim(0, np.max(true_losses)*1.1)
plt.xlabel('Iteration No.')
plt.ylabel('Mean Squared Error')
plt.title(f"p={p}, n={n}, L={L}, sigma={sigma}")
plt.legend()
plt.savefig(f"plots/{p}_{n}_{L}_{sigma:.2f}.png")
plt.show()
