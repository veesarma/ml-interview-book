"""Optimizer trajectories on an ill-conditioned quadratic and on Rosenbrock."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.optim.optimizers import SGD, Adam, Momentum, Nesterov, RMSProp, run_optimizer

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_optimizer_trajectories.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: quadratic with condition number 50.
    A = np.diag([1.0, 50.0])
    grad_q = lambda x: A @ x  # noqa: E731
    g1, g2 = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-1.2, 1.2, 200))
    F = 0.5 * (A[0, 0] * g1**2 + A[1, 1] * g2**2)
    ax = axes[0]
    ax.contour(g1, g2, F, levels=np.logspace(-2, 1.7, 14), cmap="Greys", linewidths=0.8)
    x0 = np.array([-2.5, 1.0])
    runs = [
        ("SGD lr=0.035", lambda p: SGD(p, lr=0.035)),
        ("Momentum 0.9", lambda p: Momentum(p, lr=0.01, momentum=0.9)),
        ("Nesterov 0.9", lambda p: Nesterov(p, lr=0.01, momentum=0.9)),
        ("RMSProp", lambda p: RMSProp(p, lr=0.05)),
        ("Adam", lambda p: Adam(p, lr=0.1)),
    ]
    for (name, make), c in zip(runs, ["C0", "C1", "C2", "C3", "C4"]):
        traj = run_optimizer(make([x0.copy()]), grad_q, 60)  # (61, 2)
        ax.plot(traj[:, 0], traj[:, 1], "-o", ms=2, lw=1.2, color=c, label=name)
    ax.plot(0, 0, "k*", ms=10)
    ax.set_title("κ = 50 quadratic: SGD zig-zags along the steep axis;\nadaptive methods equalise per-coordinate scale", fontsize=9)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=7, loc="lower right")

    # Right: Rosenbrock.
    def grad_rb(p):
        x, y = p
        return np.array([-2 * (1 - x) - 400 * x * (y - x**2), 200 * (y - x**2)])

    ax = axes[1]
    g1, g2 = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
    F = (1 - g1) ** 2 + 100 * (g2 - g1**2) ** 2
    ax.contour(g1, g2, np.log1p(F), levels=25, cmap="Greys", linewidths=0.8)
    x0 = np.array([-1.5, 2.0])
    runs = [
        ("SGD lr=1.5e-3", lambda p: SGD(p, lr=1.5e-3)),
        ("Momentum 0.9, lr=2e-4", lambda p: Momentum(p, lr=2e-4, momentum=0.9)),
        ("RMSProp lr=5e-3", lambda p: RMSProp(p, lr=5e-3)),
        ("Adam lr=2e-2", lambda p: Adam(p, lr=2e-2)),
    ]
    for (name, make), c in zip(runs, ["C0", "C1", "C3", "C4"]):
        traj = run_optimizer(make([x0.copy()]), grad_rb, 1500)  # (1501, 2)
        ax.plot(traj[:, 0], traj[:, 1], lw=1.4, color=c, label=name)
        ax.plot(traj[-1, 0], traj[-1, 1], "o", color=c, ms=5)
    ax.plot(1, 1, "k*", ms=10)
    ax.set_title("Rosenbrock (1500 steps, dots = final iterate): the largest stable\nSGD/momentum lr crawls along the valley; adaptive steps reach (1, 1)", fontsize=9)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1, 3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
