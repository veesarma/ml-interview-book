"""k-means: Voronoi cells of the final centroids, k-means++ seeds, and the
monotone objective."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, GREY, ORANGE, RED, save, style
from mlbook.classical.kmeans import kmeans, kmeans_plusplus_init, squared_distances

style()
rng = np.random.default_rng(0)
C_true = np.array([[0, 0], [5, 1], [1, 5], [6, 6.0]])
lab = rng.integers(0, 4, 400)
X = C_true[lab] + 0.8 * rng.standard_normal((400, 2))  # (400, 2)
seeds = kmeans_plusplus_init(X, 4, np.random.default_rng(3))
C, labels, hist = kmeans(X, 4, seed=3)
_, _, hist_rand = kmeans(X, 4, init="random", seed=3)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
g = np.linspace(-3, 9, 300)
G1, G2 = np.meshgrid(g, g)
grid = np.stack([G1.ravel(), G2.ravel()], axis=1)  # (90000, 2)
Z = squared_distances(grid, C).argmin(axis=1).reshape(G1.shape)  # (300, 300) Voronoi labels
ax.contourf(G1, G2, Z, levels=np.arange(-0.5, 4.5), cmap="Pastel1", alpha=0.9)
ax.contour(G1, G2, Z, levels=np.arange(-0.5, 4.5), colors="black", linewidths=0.6)
ax.scatter(X[:, 0], X[:, 1], s=6, color=GREY)
ax.scatter(seeds[:, 0], seeds[:, 1], marker="x", s=70, color=ORANGE, label="k-means++ seeds", linewidths=2)
ax.scatter(C[:, 0], C[:, 1], marker="*", s=180, color=RED, edgecolor="white", label="final centroids")
ax.set_title("Voronoi cells = assignment step")
ax.legend(loc="upper left"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")

ax = axes[1]
ax.plot(hist, "o-", color=BLUE, label="k-means++ init", ms=4)
ax.plot(hist_rand, "s-", color=ORANGE, label="random init", ms=4)
ax.set_xlabel("Lloyd iteration"); ax.set_ylabel("objective J = Σ ||x − μ_c||²")
ax.set_title("J never increases; k-means++ starts lower")
ax.legend()
save(fig, "part02_voronoi")
