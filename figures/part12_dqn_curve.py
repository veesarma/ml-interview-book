"""DQN vs Double+Dueling DQN learning curves on PointMass1D (3 seeds each, smoothed).

Writes docs/assets/figures/part12_dqn_curve.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.rl.dqn import DQNConfig, train_dqn  # noqa: E402
from mlbook.rl.envs import PointMass1D  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_dqn_curve.png"
torch.set_num_threads(1)
EPISODES, SEEDS = 120, 4


def smooth(x, k=10):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor="white")
    variants = {
        "DQN (target refresh every 200 steps)": dict(cfg=DQNConfig(), dueling=False),
        "no target network (refresh every step)": dict(cfg=DQNConfig(target_update_every=1), dueling=False),
        "stale target (refresh every 2000 steps)": dict(cfg=DQNConfig(target_update_every=2000), dueling=False),
    }
    for name, kw in variants.items():
        runs = np.stack([smooth(np.array(train_dqn(PointMass1D(), EPISODES, kw["cfg"], seed=s, dueling=kw["dueling"])[1])) for s in range(SEEDS)])
        x = np.arange(runs.shape[1]) + 10
        ax.plot(x, runs.mean(0), label=name, lw=1.6)
        ax.fill_between(x, runs.min(0), runs.max(0), alpha=0.15)
    ax.axhline(-16.3, color="gray", ls=":", lw=1, label="uniform random policy (≈ -16)")
    ax.axhline(-3.2, color="black", ls="--", lw=1, label="hand-tuned PD controller (≈ -3)")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return (10-episode moving average)")
    ax.set_title("Target-network period on PointMass1D (obs = (x, v), 3 actions); band = min/max over 4 seeds", fontsize=9.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(color="#eeeeee")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
