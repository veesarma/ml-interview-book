"""Eigenvectors of a covariance matrix are the axes of the data ellipse (PCA)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_covariance_eigen.png"


def main() -> None:
    rng = np.random.default_rng(0)
    theta = np.deg2rad(30)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # (2, 2)
    Sigma = R @ np.diag([4.0, 0.5]) @ R.T  # (2, 2) rotated anisotropic covariance
    X = rng.multivariate_normal(np.zeros(2), Sigma, size=600)  # (600, 2)
    S = np.cov(X, rowvar=False)  # (2, 2) sample covariance
    lam, V = np.linalg.eigh(S)  # (2,), (2, 2) ascending
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.4, color="C0", label="samples")
    for k in (1, 2):
        w, h = 2 * k * np.sqrt(lam)  # k-sigma ellipse axes
        ang = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        ax.add_patch(Ellipse((0, 0), w, h, angle=ang, fill=False, color="C3", lw=1.5, ls="--" if k == 2 else "-"))
    for j, c in enumerate(["C1", "C2"]):
        vec = V[:, j] * np.sqrt(lam[j]) * 2
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2.5, color=c))
        ax.text(vec[0] * 1.1, vec[1] * 1.1, f"v{j+1}, λ{j+1}={lam[j]:.2f}", color=c, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.set_title("Covariance eigenvectors = principal axes; ellipse = 1σ and 2σ")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend(loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
