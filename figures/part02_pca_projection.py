"""PCA: principal directions on correlated 2-D data, and the explained-variance
spectrum of a 20-D dataset with a 3-D latent structure."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, GREY, ORANGE, save, style
from mlbook.classical.pca import inverse_transform, pca_svd, transform

style()
rng = np.random.default_rng(0)
A = np.array([[2.0, 0.8], [0.4, 0.6]])
X = rng.standard_normal((300, 2)) @ A.T + np.array([1.0, 2.0])  # (300, 2)
V, var, mu = pca_svd(X, 2)
X1 = inverse_transform(transform(X, V[:, :1], mu), V[:, :1], mu)  # (300, 2) rank-1 reconstruction

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=8, color=GREY, label="data")
ax.scatter(X1[:, 0], X1[:, 1], s=6, color=ORANGE, label="projection onto PC1")
for j, (c, lbl) in enumerate([(BLUE, "PC1"), (ORANGE, "PC2")]):
    d = V[:, j] * 2.5 * np.sqrt(var[j])
    ax.annotate("", xy=mu + d, xytext=mu, arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
    ax.text(*(mu + d * 1.1), lbl, color=c, fontweight="bold")
ax.set_aspect("equal", adjustable="datalim"); ax.set_title("Principal directions = covariance eigenvectors")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(loc="upper left")

ax = axes[1]
latent = rng.standard_normal((500, 3)) * np.array([4.0, 2.0, 1.0])
Xh = latent @ rng.standard_normal((3, 20)) + 0.5 * rng.standard_normal((500, 20))  # (500, 20)
_, var20, _ = pca_svd(Xh, 20)
ratio = var20 / var20.sum()
ax.bar(np.arange(1, 21), ratio, color=BLUE, width=0.7, label="explained variance ratio")
ax.plot(np.arange(1, 21), np.cumsum(ratio), "o-", color=ORANGE, ms=3, label="cumulative")
ax.set_xlabel("component"); ax.set_ylabel("fraction of variance")
ax.set_title("Scree plot: 3 latent factors + isotropic noise")
ax.legend()
save(fig, "part02_pca_projection")
