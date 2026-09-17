"""Finite-difference gradient error vs step size (truncation vs round-off) and
the gradient as the direction of steepest ascent on a contour plot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_gradient_check.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # Left: contour of f(x) = 1/2 x^T A x with gradient arrows (perpendicular to contours).
    ax = axes[0]
    A = np.array([[3.0, 1.0], [1.0, 1.0]])
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
    P = np.stack([g1, g2], axis=-1)  # (200, 200, 2)
    F = 0.5 * np.einsum("...i,ij,...j->...", P, A, P)  # (200, 200)  x^T A x per grid point
    ax.contour(g1, g2, F, levels=12, cmap="Greys")
    pts = np.array([[1.0, 0.5], [-1.0, 1.2], [0.5, -1.5], [-1.2, -0.8]])
    for p in pts:
        g = A @ p
        g = 0.5 * g / np.linalg.norm(g)
        ax.annotate("", xy=p + g, xytext=p, arrowprops=dict(arrowstyle="->", lw=2, color="C3"))
    ax.set_aspect("equal")
    ax.set_title("∇f is normal to level sets (steepest ascent)")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    # Right: error of forward vs central differences against eps.
    ax = axes[1]
    f = lambda x: np.sin(x) * np.exp(x / 3)  # noqa: E731
    df = lambda x: np.cos(x) * np.exp(x / 3) + np.sin(x) * np.exp(x / 3) / 3  # noqa: E731
    x0 = 1.0
    eps = np.logspace(-12, 0, 200)
    fwd = np.abs((f(x0 + eps) - f(x0)) / eps - df(x0))
    ctr = np.abs((f(x0 + eps) - f(x0 - eps)) / (2 * eps) - df(x0))
    ax.loglog(eps, fwd, label="forward difference  O(ε)", color="C0")
    ax.loglog(eps, ctr, label="central difference  O(ε²)", color="C1")
    ax.axvline(1e-6, color="0.6", ls="--", lw=1)
    ax.text(1.3e-6, 1e-2, "ε ≈ 1e-6 sweet spot", fontsize=9, color="0.4")
    ax.set_xlabel("ε")
    ax.set_ylabel("|numerical − analytic|")
    ax.set_title("Truncation error (right) vs round-off (left)")
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
