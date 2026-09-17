"""How much labelled data you have decides the method, and what each family costs.

Writes docs/assets/figures/part10_label_budget.png.  Run:  python figures/part10_label_budget.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part10_label_budget.png"

REGIMES = [
    (0, 1.1, "0 labels", "SSL pretraining\n(DINOv2, MAE, CLIP)\nthen zero-shot or\nk-NN on features", 0),
    (1.1, 2.6, "$10^1$ to $10^3$", "Frozen SSL features\n+ linear probe.\nActive learning to\nspend the next labels", 1),
    (2.6, 4.1, "$10^3$ to $10^5$", "Fine-tune an SSL\nbackbone. FixMatch or\npseudo-labelling on the\nunlabelled pool", 2),
    (4.1, 5.6, "$10^5$ to $10^7$", "Weak supervision and\nauto-labelling to reach\nvolume. Noisy Student\nself-training", 3),
    (5.6, 7.0, "$> 10^7$", "Supervised training wins.\nSpend the budget on\nlabel quality and on\nmining hard cases", 4),
]


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.0), facecolor="white",
                                  gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax.set_xlim(0, 7); ax.set_ylim(0, 1); ax.axis("off")
    for x0, x1, label, body, ci in REGIMES:
        ax.add_patch(FancyBboxPatch((x0 + 0.05, 0.12), x1 - x0 - 0.1, 0.62,
                                    boxstyle="round,pad=0.02", linewidth=1.3,
                                    edgecolor=colors[ci], facecolor=colors[ci], alpha=0.12))
        ax.text((x0 + x1) / 2, 0.70, label, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=colors[ci])
        ax.text((x0 + x1) / 2, 0.60, body, ha="center", va="top", fontsize=8.2, color="#222222")
    ax.annotate("", xy=(7.0, 0.05), xytext=(0.0, 0.05),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color="#555555"))
    ax.text(3.5, 0.005, "labelled examples for your task", ha="center", fontsize=9, color="#555555")
    ax.text(3.5, 0.93, "The decision is driven by label count, not by taste",
            ha="center", fontsize=12, fontweight="bold")

    # right panel: the shape of the curve everyone is trying to move
    n = np.logspace(1, 6, 100)
    sup = 1.0 - 0.72 * (np.log10(n) - 1) / 5
    ssl = 1.0 - 0.72 * (np.log10(n * 30) - 1) / 5
    ssl = np.clip(ssl, 0.18, 1.0)
    ax2.plot(n, sup, color=colors[0], linewidth=1.8, label="train from scratch")
    ax2.plot(n, ssl, color=colors[2], linewidth=1.8, label="fine-tune an SSL backbone")
    ax2.fill_between(n, ssl, sup, color=colors[2], alpha=0.12)
    ax2.set_xscale("log")
    ax2.set_xlabel("labelled examples (log scale)")
    ax2.set_ylabel("error (arbitrary units)")
    ax2.set_title("Pretraining shifts the curve left, it does not change its slope", fontsize=10)
    ax2.annotate("the gap closes\nas labels grow", xy=(3e5, 0.27), xytext=(1.2e4, 0.13),
                 fontsize=8.5, color="#555555",
                 arrowprops=dict(arrowstyle="->", color="#888888", linewidth=1.0))
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.text(0.02, -0.22, "Schematic: the shape is the reproducible finding (SimCLR, MAE, DINOv2, Noisy Student),\nthe units are not measurements.",
             transform=ax2.transAxes, fontsize=7.5, color="#777777")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
