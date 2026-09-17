"""Cumulative regret of epsilon-greedy, UCB1 and Thompson sampling on a 4-armed Bernoulli bandit (20 seeds).

Writes docs/assets/figures/part12_bandit_regret.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.rl.bandits import BernoulliBandit, EpsilonGreedy, ThompsonBeta, UCB1, run_bandit  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_bandit_regret.png"
PROBS = np.array([0.1, 0.5, 0.6, 0.9])
HORIZON, SEEDS = 5000, 20


def main() -> None:
    solvers = {
        "ε-greedy (ε=0.1)": lambda rng: EpsilonGreedy(4, 0.1, rng),
        "ε-greedy (ε=0.01)": lambda rng: EpsilonGreedy(4, 0.01, rng),
        "UCB1 (c=√2)": lambda rng: UCB1(4),
        "Thompson (Beta)": lambda rng: ThompsonBeta(4, rng),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="white")
    for name, make in solvers.items():
        curves = np.stack([run_bandit(BernoulliBandit(PROBS, np.random.default_rng(s)), make(np.random.default_rng(s)), HORIZON) for s in range(SEEDS)])  # (SEEDS, T)
        mean, sd = curves.mean(0), curves.std(0)
        for ax in axes:
            ax.plot(mean, label=name, lw=1.6)
            ax.fill_between(np.arange(HORIZON), mean - sd, mean + sd, alpha=0.12)
    axes[0].set_title("Cumulative regret (mean ± 1 sd over 20 seeds)", fontsize=9.5, loc="left")
    axes[1].set_xscale("log")
    axes[1].set_title("Same, log-x: UCB/Thompson bend over (O(log T)), ε-greedy stays linear", fontsize=9.5, loc="left")
    for ax in axes:
        ax.set_xlabel("pulls t")
        ax.set_ylabel("regret")
        ax.grid(color="#eeeeee")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
