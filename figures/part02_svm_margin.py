"""SVM: maximum-margin hyperplane with support vectors (linear), and an RBF
kernel boundary on non-separable data."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, GREY, ORANGE, RED, save, style
from mlbook.classical.kernels import linear_kernel, rbf_kernel
from mlbook.classical.svm import SVM

style()
rng = np.random.default_rng(0)
n = 80
X = rng.standard_normal((n, 2))
y = np.where(X[:, 0] + 0.6 * X[:, 1] > 0, 1, -1)
X = X + 0.7 * y[:, None] * np.array([1.0, 0.6])  # (n, 2) widen the margin
svm = SVM(linear_kernel, C=10.0, max_passes=40).fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
g = np.linspace(-4, 4, 200)
G1, G2 = np.meshgrid(g, g)
grid = np.stack([G1.ravel(), G2.ravel()], axis=1)
Z = svm.decision_function(grid).reshape(G1.shape)  # (200, 200)
ax.contour(G1, G2, Z, levels=[-1, 0, 1], colors=["black"] * 3, linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1])
ax.scatter(X[y == -1, 0], X[y == -1, 1], s=16, color=BLUE, label="y = −1")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, color=ORANGE, label="y = +1")
sv = svm.support_
ax.scatter(X[sv, 0], X[sv, 1], s=90, facecolor="none", edgecolor=RED, linewidth=1.5, label=f"support vectors ({len(sv)})")
ax.set_title("Linear SVM: f = 0 (solid), f = ±1 (dashed)")
ax.legend(loc="upper left"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")

ax = axes[1]
m = 200
Xr = rng.uniform(-2, 2, (m, 2))
yr = np.where(Xr[:, 0] ** 2 + Xr[:, 1] ** 2 < 1.5, 1, -1)
flip = rng.random(m) < 0.05
yr[flip] = -yr[flip]
svm_r = SVM(lambda A, B: rbf_kernel(A, B, gamma=1.0), C=2.0, max_passes=20).fit(Xr, yr)
g = np.linspace(-2.2, 2.2, 200)
G1, G2 = np.meshgrid(g, g)
Zr = svm_r.decision_function(np.stack([G1.ravel(), G2.ravel()], axis=1)).reshape(G1.shape)
ax.contourf(G1, G2, Zr, levels=[-10, 0, 10], colors=[BLUE, ORANGE], alpha=0.15)
ax.contour(G1, G2, Zr, levels=[-1, 0, 1], colors=["black"] * 3, linestyles=["--", "-", "--"], linewidths=[0.8, 1.6, 0.8])
ax.scatter(Xr[yr == -1, 0], Xr[yr == -1, 1], s=12, color=BLUE)
ax.scatter(Xr[yr == 1, 0], Xr[yr == 1, 1], s=12, color=ORANGE)
ax.scatter(Xr[svm_r.support_, 0], Xr[svm_r.support_, 1], s=60, facecolor="none", edgecolor=RED, linewidth=1)
ax.set_title(f"RBF SVM (γ=1, C=2): {len(svm_r.support_)}/{m} support vectors")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
save(fig, "part02_svm_margin")
