"""Optimal state values (heatmap) and greedy policy (arrows) of the 4x4 GridWorld from value iteration.

Writes docs/assets/figures/part12_gridworld_values.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.rl.dynamic_programming import policy_evaluation, value_iteration  # noqa: E402
from mlbook.rl.envs import GRID_MOVES, GridWorld  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_gridworld_values.png"


def draw(ax, env, V, pi, title):
    grid = V.reshape(env.n_rows, env.n_cols)  # (rows, cols)
    im = ax.imshow(grid, cmap="viridis", vmin=-1, vmax=1)
    for s in range(env.n_states):
        r, c = env.to_cell(s)
        if (r, c) in env.walls:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#444444"))
            continue
        if (r, c) == env.goal:
            ax.text(c, r, "G\n+1", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            continue
        if (r, c) == env.pit:
            ax.text(c, r, "pit\n-1", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            continue
        dr, dc = GRID_MOVES[pi[s]]
        ax.annotate("", xy=(c + 0.32 * dc, r + 0.32 * dr), xytext=(c - 0.1 * dc, r - 0.1 * dr),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6))
        ax.text(c, r + 0.4, f"{V[s]:.2f}", ha="center", va="center", color="white", fontsize=7.5)
    ax.text(env.start[1], env.start[0] - 0.38, "start", ha="center", va="center", color="white", fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, loc="left")
    return im


def main() -> None:
    env = GridWorld()
    gamma = 0.95
    V_star, pi_star, n_iters = value_iteration(env.P, env.R, gamma)
    V_rand = policy_evaluation(env.P, env.R, np.full((16, 4), 0.25), gamma)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), facecolor="white")
    draw(axes[0], env, V_rand, np.argmax(env.R + gamma * env.P @ V_rand, axis=1),
         "V^π, uniform random policy\n(arrows: one greedy step)")
    im = draw(axes[1], env, V_star, pi_star,
              f"V* and π*, value iteration\n({n_iters} sweeps, γ=0.95, slip=0.1)")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="value")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
