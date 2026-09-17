"""Logistic regression: decision boundary with probability shading, and a
reliability diagram before/after Platt scaling of an over-confident score."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, GREY, ORANGE, save, style
from mlbook.classical.logistic_regression import fit_logistic_newton, platt_scaling, sigmoid

style()
rng = np.random.default_rng(0)
N = 300
y = rng.integers(0, 2, N)
X = np.array([[-1.2, -0.8], [1.2, 0.8]])[y] + rng.standard_normal((N, 2))  # (N, 2)
Xb = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, 3)
w = fit_logistic_newton(Xb, y.astype(float), l2=1e-3)  # (3,)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
g = np.linspace(-4, 4, 200)
G1, G2 = np.meshgrid(g, g)  # (200, 200)
P = sigmoid(np.stack([G1.ravel(), G2.ravel(), np.ones(G1.size)], axis=1) @ w).reshape(G1.shape)  # (200, 200)
cs = ax.contourf(G1, G2, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
ax.contour(G1, G2, P, levels=[0.5], colors="black", linewidths=1.5)
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=14, color=BLUE, label="y = 0", edgecolor="white", linewidth=0.4)
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=14, color=ORANGE, label="y = 1", edgecolor="white", linewidth=0.4)
fig.colorbar(cs, ax=ax, label="P(y = 1 | x)")
ax.set_title("Decision boundary is the level set σ(w·x + b) = 0.5")
ax.legend(loc="upper left")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")

# Calibration: an over-confident scorer (logits scaled ×3) fixed by Platt scaling.
ax = axes[1]
M = 4000
y2 = rng.integers(0, 2, M)
X2 = np.array([[-1.2, -0.8], [1.2, 0.8]])[y2] + rng.standard_normal((M, 2))
z_true = np.concatenate([X2, np.ones((M, 1))], axis=1) @ w  # (M,) roughly calibrated logits
z_over = 3.0 * z_true  # (M,) over-confident
a, b = platt_scaling(z_over, y2.astype(float))
bins = np.linspace(0, 1, 11)


def reliability(p):
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    conf = np.array([p[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)])
    acc = np.array([y2[idx == i].mean() if (idx == i).any() else np.nan for i in range(10)])
    return conf, acc


for p, label, color in [(sigmoid(z_over), "over-confident scorer", ORANGE), (sigmoid(a * z_over + b), "after Platt scaling", BLUE)]:
    conf, acc = reliability(p)
    ax.plot(conf, acc, "o-", color=color, label=label, ms=5, lw=1.5)
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect calibration")
ax.set_xlabel("mean predicted probability (bin)"); ax.set_ylabel("empirical positive rate")
ax.set_title(f"Reliability diagram; Platt fit a = {a:.2f}, b = {b:.2f}")
ax.legend(loc="upper left")
save(fig, "part02_logistic_boundary_calibration")
