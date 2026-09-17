"""GMM fitted by EM: density contours, soft responsibilities, and the
monotone log-likelihood."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, GREY, ORANGE, save, style
from mlbook.classical.gmm import GMM, log_gaussian

style()
rng = np.random.default_rng(1)
n = 600
z = rng.random(n) < 0.6
X = np.where(
    z[:, None],
    rng.multivariate_normal([2.0, 1.0], [[1.2, 0.9], [0.9, 1.0]], n),
    rng.multivariate_normal([-2.0, -0.5], [[0.7, -0.4], [-0.4, 0.8]], n),
)  # (n, 2)
gmm = GMM(k=2, n_iters=100, seed=0).fit(X)
R = gmm.predict_proba(X)  # (n, 2)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.35})
ax = axes[0]
g1 = np.linspace(-5, 5.5, 200); g2 = np.linspace(-4, 4.5, 200)
G1, G2 = np.meshgrid(g1, g2)
grid = np.stack([G1.ravel(), G2.ravel()], axis=1)  # (40000, 2)
dens = sum(gmm.pi[j] * np.exp(log_gaussian(grid, gmm.mu[j], gmm.Sigma[j])) for j in range(2)).reshape(G1.shape)
ax.contour(G1, G2, dens, levels=8, colors=GREY, linewidths=0.8)
sc = ax.scatter(X[:, 0], X[:, 1], c=R[:, 1], cmap="coolwarm", s=8, vmin=0, vmax=1)
ax.scatter(gmm.mu[:, 0], gmm.mu[:, 1], marker="*", s=200, color="black", edgecolor="white")
fig.colorbar(sc, ax=ax, label="responsibility r_i2")
ax.set_title("Mixture density contours; colour = soft assignment"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax = axes[1]
ax.plot(gmm.history_, "o-", color=BLUE, ms=3)
ax.set_xlabel("EM iteration"); ax.set_ylabel("log-likelihood")
ax.set_title("EM: log-likelihood is non-decreasing")
save(fig, "part02_gmm_contours")
