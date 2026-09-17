"""GRPO group-relative advantages: G = 8 sampled responses to one prompt, their binary
verifier rewards, and the normalised advantages (r - mean) / std that replace a critic.
Writes docs/assets/figures/part07_grpo_advantages.png.
Run from the repository root:  python figures/part07_grpo_advantages.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part07_grpo_advantages.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), facecolor="white")
    groups = {
        "3 of 8 correct (useful group)": np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=float),
        "1 of 8 correct (rare success, large A)": np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=float),
        "8 of 8 correct (zero gradient)": np.ones(8),
    }
    for ax, (title, r) in zip(axes, groups.items()):
        mean, std = r.mean(), r.std()
        adv = (r - mean) / (std + 1e-6) if std > 0 else np.zeros_like(r)
        idx = np.arange(len(r))
        ax.bar(idx - 0.18, r, width=0.36, color=colors[0], label="reward $r_i$")
        ax.bar(idx + 0.18, adv, width=0.36, color=colors[1], label="advantage $\\hat A_i$")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(mean, color=colors[0], linewidth=0.8, linestyle="--")
        ax.text(7.4, mean + 0.05, f"mean = {mean:.2f}", fontsize=7.5, ha="right", color=colors[0])
        ax.set_xticks(idx)
        ax.set_xticklabels([f"$y_{i+1}$" for i in idx], fontsize=8)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_ylim(-1.6, 2.9)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[0].set_ylabel("value")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
