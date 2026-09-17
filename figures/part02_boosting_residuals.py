"""Gradient boosting as functional gradient descent: each round fits the
negative gradient (the residual for squared loss)."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, GREY, ORANGE, RED, save, style
from mlbook.classical.gradient_boosting import GradientBoostedTrees

style()
rng = np.random.default_rng(0)
N = 200
X = np.sort(rng.uniform(-3, 3, (N, 1)), axis=0)  # (N, 1)
y = np.sin(X[:, 0]) + 0.3 * X[:, 0] + 0.2 * rng.standard_normal(N)  # (N,)
gbt = GradientBoostedTrees(n_rounds=30, learning_rate=0.3, max_depth=2, lam=0.0).fit(X, y)

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
F = np.full(N, gbt.base_score)  # (N,) running prediction
rounds = [0, 1, 2, 5, 10, 30]
step = 0
for ax, m in zip(axes.ravel(), rounds):
    while step < m:
        F = F + gbt.lr * gbt.trees[step].predict(X)
        step += 1
    resid = y - F  # (N,) negative gradient of ½(F-y)²
    ax.scatter(X[:, 0], y, s=8, color=GREY, label="data")
    ax.plot(X[:, 0], F, color=BLUE, lw=2, label=f"$F_{{{m}}}$")
    ax.vlines(X[::6, 0], F[::6], y[::6], color=ORANGE, lw=0.8, alpha=0.8, label="residual = −∂L/∂F")
    ax.set_title(f"round {m}: train MSE {np.mean(resid**2):.3f}")
    ax.set_xlabel("x")
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Each round fits a depth-2 tree to the residuals and takes a step of size η = 0.3 in function space", fontsize=10)
fig.tight_layout()
save(fig, "part02_boosting_residuals")
