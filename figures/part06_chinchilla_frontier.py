"""Iso-loss contours of L(N, D) = E + A/N^alpha + B/D^beta with the compute-optimal
frontier N*(C) overlaid, plus the iso-FLOP curves that Chinchilla's approach 2 fits.

Writes docs/assets/figures/part06_chinchilla_frontier.png.  Run from the repo root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.llm.scaling_laws import chinchilla_loss, compute_optimal, loss_at_fixed_compute  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_chinchilla_frontier.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")

    # Left: iso-loss contours in (N, D) log-log space with the compute-optimal frontier.
    N = np.logspace(7, 12, 200)  # (200,) parameters
    D = np.logspace(8, 14, 200)  # (200,) tokens
    NN, DD = np.meshgrid(N, D)  # (200, 200) each
    L = chinchilla_loss(NN, DD)  # (200, 200)
    cs = ax1.contour(NN, DD, L, levels=[1.9, 2.0, 2.1, 2.2, 2.4, 2.6, 3.0], colors="#999999", linewidths=0.8)
    ax1.clabel(cs, fmt="L=%.1f", fontsize=7)
    Cs = np.logspace(18, 26, 50)
    frontier = np.array([compute_optimal(c) for c in Cs])  # (50, 2)
    ax1.plot(frontier[:, 0], frontier[:, 1], color=colors[0], linewidth=2, label="compute-optimal $N^*(C), D^*(C)$")
    ax1.plot(N, 20 * N, "--", color=colors[1], linewidth=1.2, label="D = 20 N")
    for name, n, d in [("Chinchilla", 70e9, 1.4e12), ("Llama-3-8B", 8e9, 15e12), ("Llama-3-70B", 70e9, 15e12)]:
        ax1.scatter([n], [d], color=colors[3], zorder=5, s=22)
        ax1.annotate(name, (n, d), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("parameters N")
    ax1.set_ylabel("training tokens D")
    ax1.set_title("Iso-loss contours and the compute-optimal frontier", fontsize=10, loc="left")
    ax1.legend(fontsize=8, frameon=False, loc="upper left")
    ax1.grid(color="#eeeeee", linewidth=0.5)

    # Right: iso-FLOP curves (loss vs N at fixed C); the minimum of each is one point of the frontier.
    for i, C in enumerate([1e19, 1e20, 1e21, 1e22, 1e23]):
        n_grid = np.logspace(7.5, 11.5, 300)
        losses = loss_at_fixed_compute(C, n_grid)
        ax2.plot(n_grid, losses, color=colors[i], linewidth=1.6, label=f"C = 1e{int(np.log10(C))}")
        n_star, _ = compute_optimal(C)
        ax2.scatter([n_star], [loss_at_fixed_compute(C, np.array([n_star]))[0]], color=colors[i], s=20, zorder=5)
    ax2.set_xscale("log")
    ax2.set_ylim(1.9, 3.6)
    ax2.set_xlabel("parameters N  (tokens D = C / 6N)")
    ax2.set_ylabel("predicted loss")
    ax2.set_title("Iso-FLOP curves: each minimum is one frontier point", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(color="#eeeeee", linewidth=0.5)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
