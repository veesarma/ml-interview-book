"""Mid-training schedule: learning rate and data mixture over a stable phase, an
annealing phase on up-weighted high-quality data, and a long-context extension stage.

Writes docs/assets/figures/part06_midtraining_pipeline.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_midtraining_pipeline.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    steps = np.linspace(0, 100, 1000)  # (1000,) percent of training
    warm, anneal_start, ctx_start = 2.0, 80.0, 92.0
    lr = np.where(steps < warm, steps / warm, 1.0)
    lr = np.where((steps >= anneal_start) & (steps < ctx_start), 1.0 - (steps - anneal_start) / (ctx_start - anneal_start) * 0.9, lr)
    lr = np.where(steps >= ctx_start, 0.1, lr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), facecolor="white", sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax1.plot(steps, lr, color=colors[0], linewidth=2)
    ax1.set_ylabel("learning rate (relative)")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("A mid-training schedule: stable → anneal on high-quality mix → long-context stage", fontsize=10, loc="left")

    web = np.where(steps < anneal_start, 0.70, np.where(steps < ctx_start, 0.40, 0.35))
    code = np.where(steps < anneal_start, 0.15, np.where(steps < ctx_start, 0.20, 0.15))
    math = np.where(steps < anneal_start, 0.05, np.where(steps < ctx_start, 0.15, 0.10))
    hq = np.where(steps < anneal_start, 0.10, np.where(steps < ctx_start, 0.25, 0.15))
    longdoc = np.where(steps < ctx_start, 0.0, 0.25)
    ax2.stackplot(steps, web, code, math, hq, longdoc, labels=["web (filtered)", "code", "math", "curated / synthetic high-quality", "long documents"],
                  colors=[colors[7], colors[0], colors[1], colors[2], colors[4]], alpha=0.85)
    ax2.set_ylabel("data mixture (fraction of tokens)")
    ax2.set_xlabel("percent of pretraining tokens")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    for ax in (ax1, ax2):
        ax.axvline(anneal_start, color="#888888", linewidth=0.8, linestyle="--")
        ax.axvline(ctx_start, color="#888888", linewidth=0.8, linestyle="--")
        ax.grid(color="#eeeeee", linewidth=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.text(anneal_start + 0.5, 1.05, "anneal", fontsize=8, color="#555555")
    ax1.text(ctx_start + 0.5, 1.05, "context ext.", fontsize=8, color="#555555")
    ax1.text(40, 1.05, "stable phase (constant LR, broad mixture)", fontsize=8, color="#555555", ha="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
