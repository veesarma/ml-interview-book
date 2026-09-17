"""A gallery of the distributions every ML engineer should recognise on sight,
plus conditioning a 2-D Gaussian."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.math.probability import beta_logpdf, gaussian_condition, poisson_logpmf

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_distributions.png"


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    ax = axes[0, 0]
    x = np.linspace(0.005, 0.995, 400)
    for a, b in [(1, 1), (2, 5), (5, 2), (20, 20), (0.5, 0.5)]:
        ax.plot(x, np.exp(beta_logpdf(x, a, b)), label=f"Beta({a},{b})")
    ax.set_title("Beta: a prior / posterior over a probability")
    ax.set_ylim(0, 6)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    k = np.arange(0, 25)
    for lam in [1, 4, 10]:
        ax.stem(k + (lam - 4) * 0.12, np.exp(poisson_logpmf(k, lam)), linefmt=f"C{[1,4,10].index(lam)}-", markerfmt=f"C{[1,4,10].index(lam)}o", basefmt=" ", label=f"Poisson(λ={lam})")
    ax.set_title("Poisson: counts per interval (mean = variance = λ)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    alphas = [np.array([1.0, 1.0, 1.0]), np.array([10.0, 10.0, 10.0]), np.array([0.3, 0.3, 0.3])]
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])  # (3, 2) simplex corners in 2-D
    for i, alpha in enumerate(alphas):
        P = rng.dirichlet(alpha, size=300)  # (300, 3)
        XY = P @ corners  # (300, 2) barycentric -> planar
        ax.scatter(XY[:, 0] + i * 1.2, XY[:, 1], s=6, alpha=0.6, color=f"C{i}")
        tri = np.vstack([corners, corners[:1]]) + np.array([i * 1.2, 0])
        ax.plot(tri[:, 0], tri[:, 1], color="0.5", lw=1)
        ax.text(0.5 + i * 1.2, -0.12, f"Dir(α = {alpha[0]:g})", ha="center", fontsize=9, color=f"C{i}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_ylim(-0.2, 1.0)
    ax.set_title("Dirichlet on the simplex: α<1 sparse, α=1 flat, α≫1 peaked")

    ax = axes[1, 1]
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
    X = rng.multivariate_normal(mu, Sigma, size=1500)  # (1500, 2)
    ax.scatter(X[:, 0], X[:, 1], s=5, alpha=0.3, color="C0", label="joint p(x₁, x₂)")
    x2_obs = 1.5
    ax.axhline(x2_obs, color="C3", lw=1.5, ls="--", label=f"observe x₂ = {x2_obs}")
    m, S = gaussian_condition(mu, Sigma, np.array([0]), np.array([1]), np.array([x2_obs]))
    grid = np.linspace(-3.5, 3.5, 300)
    pdf = np.exp(-0.5 * (grid - m[0]) ** 2 / S[0, 0]) / np.sqrt(2 * np.pi * S[0, 0])
    ax.plot(grid, x2_obs + pdf * 1.2, color="C1", lw=2, label=f"p(x₁ | x₂): μ={m[0]:.2f}, σ²={S[0,0]:.2f}")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.set_title("Conditioning a Gaussian (Schur complement)")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
