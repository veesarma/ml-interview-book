"""The PPO clipped surrogate as a function of the probability ratio, for positive and negative advantage.

Writes docs/assets/figures/part12_ppo_clip.png. Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part12_ppo_clip.png"
EPS = 0.2


def main() -> None:
    r = np.linspace(0.4, 1.6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor="white")
    for ax, A in zip(axes, (1.0, -1.0)):
        unclipped = r * A
        clipped = np.clip(r, 1 - EPS, 1 + EPS) * A
        objective = np.minimum(unclipped, clipped)
        ax.plot(r, unclipped, ls=":", color="gray", lw=1.2, label="r·A (unclipped surrogate)")
        ax.plot(r, clipped, ls="--", color="#888888", lw=1.2, label="clip(r,1-ε,1+ε)·A")
        ax.plot(r, objective, lw=2.2, label="L^CLIP = min(·,·)")
        ax.axvline(1 - EPS, color="#cccccc", lw=0.8)
        ax.axvline(1 + EPS, color="#cccccc", lw=0.8)
        ax.axvline(1.0, color="#cccccc", lw=0.8, ls="--")
        ax.scatter([1.0], [A], color="black", zorder=5, s=18)
        flat = "r > 1+ε: gradient 0\n(no incentive to push further)" if A > 0 else "r < 1-ε: gradient 0\n(no incentive to push further)"
        ax.annotate(flat, xy=(1 + EPS + 0.15 if A > 0 else 1 - EPS - 0.15, A), fontsize=7.5, ha="center", va="bottom" if A > 0 else "top", color="#333333")
        ax.set_title(f"A = {A:+.0f}: {'raise' if A > 0 else 'lower'} π(a|s), but only up to the clip", fontsize=9.5, loc="left")
        ax.set_xlabel("ratio r = π_θ(a|s) / π_old(a|s)")
        ax.set_ylabel("objective")
        ax.grid(color="#eeeeee")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
