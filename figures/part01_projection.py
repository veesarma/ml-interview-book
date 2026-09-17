"""Least squares as orthogonal projection onto the column space of X."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_projection.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    # Left: projection of a vector onto a line.
    ax = axes[0]
    u = np.array([2.0, 1.0])
    v = np.array([1.0, 2.5])
    p = (u @ v) / (u @ u) * u
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.plot([-1, 3.5], [-0.5, 1.75], color="0.6", lw=1, ls="--", label="span{u}")
    ax.annotate("", xy=u, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="C0"))
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="C1"))
    ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=2, color="C2"))
    ax.annotate("", xy=v, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.5, color="C3", ls="--"))
    ax.text(u[0] + 0.05, u[1] - 0.25, "u", color="C0", fontsize=12)
    ax.text(v[0] + 0.05, v[1] + 0.05, "v", color="C1", fontsize=12)
    ax.text(p[0] + 0.1, p[1] - 0.35, r"proj$_u$(v) = (u·v / u·u) u", color="C2", fontsize=10)
    ax.text((p[0] + v[0]) / 2 + 0.1, (p[1] + v[1]) / 2, "residual ⊥ u", color="C3", fontsize=10)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(-0.7, 3)
    ax.set_aspect("equal")
    ax.set_title("Projection onto a line")

    # Right: least squares in R^3: y projected onto plane col(X).
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    axes[1].remove()
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # columns span the xy-plane
    y = np.array([1.5, 1.0, 2.0])
    w = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat = X @ w
    gx, gy = np.meshgrid(np.linspace(0, 2.2, 5), np.linspace(0, 2.2, 5))
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.15, color="C0")
    ax.quiver(0, 0, 0, *y, color="C1", arrow_length_ratio=0.1, lw=2)
    ax.quiver(0, 0, 0, *y_hat, color="C2", arrow_length_ratio=0.1, lw=2)
    ax.plot([y_hat[0], y[0]], [y_hat[1], y[1]], [y_hat[2], y[2]], color="C3", ls="--", lw=1.5)
    ax.text(*y, "y", color="C1", fontsize=12)
    ax.text(*(y_hat + np.array([0, 0, -0.35])), "Xw* = P y", color="C2", fontsize=10)
    ax.text(*((y + y_hat) / 2 + np.array([0.15, 0, 0])), "y − Xw* ⊥ col(X)", color="C3", fontsize=9)
    ax.text(1.6, 1.8, 0.05, "col(X)", color="C0", fontsize=10)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(0, 2.2)
    ax.set_zlim(0, 2.2)
    ax.set_title("Least squares = projection onto col(X)")
    ax.view_init(elev=22, azim=-60)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
