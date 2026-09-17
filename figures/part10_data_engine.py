"""The auto-labelling flywheel, and where the cost per label actually goes.

Writes docs/assets/figures/part10_data_engine.png.  Run:  python figures/part10_data_engine.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part10_data_engine.png"

STAGES = [
    ("Fleet / product\ncollects raw data", "no labels, unbounded volume"),
    ("Offline auto-labeller\n(big model, full sequence,\nno latency budget)", "tracking, geometry, future frames"),
    ("Human verification\non the uncertain slice", "the only human cost in the loop"),
    ("Train the online model\n(small, real-time)", "distil the offline labels"),
    ("Mine failures in\nshadow mode", "disagreement, low confidence, rare classes"),
]


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.0), facecolor="white",
                                  gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35); ax.axis("off")
    n = len(STAGES)
    positions = []
    for i, (title, sub) in enumerate(STAGES):
        angle = math.pi / 2 - 2 * math.pi * i / n
        x, y = 0.92 * math.cos(angle), 0.92 * math.sin(angle)
        positions.append((x, y))
        ax.add_patch(FancyBboxPatch((x - 0.33, y - 0.16), 0.66, 0.32, boxstyle="round,pad=0.02",
                                    linewidth=1.3, edgecolor=colors[i], facecolor=colors[i], alpha=0.12))
        ax.text(x, y + 0.045, title, ha="center", va="center", fontsize=8.0, color="#111111")
        ax.text(x, y - 0.105, sub, ha="center", va="center", fontsize=6.8, color="#666666", style="italic")
    for i in range(n):
        x0, y0 = positions[i]
        x1, y1 = positions[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        sx, sy = x0 + 0.34 * dx / norm * 1.0, y0 + 0.34 * dy / norm * 1.0
        ex, ey = x1 - 0.36 * dx / norm * 1.0, y1 - 0.36 * dy / norm * 1.0
        ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.2, color="#888888",
                                     connectionstyle="arc3,rad=0.12"))
    ax.text(0, 0.06, "data engine", ha="center", fontsize=12, fontweight="bold", color="#333333")
    ax.text(0, -0.06, "each turn makes the next\nround of labels cheaper", ha="center",
            fontsize=8, color="#666666")

    # right: cost per label under three strategies, as a function of volume
    volume = np.logspace(3, 7, 60)
    human = 0.50 * np.ones_like(volume)                        # flat unit cost, per label
    engine_fixed = 2.0e5                                       # one-off build cost, in label-equivalents
    engine = 0.50 * 0.12 + engine_fixed * 0.5 / volume         # verification of a small slice + amortised build
    synth = 0.50 * 0.02 + 6.0e5 * 0.5 / volume                 # synthetic generator: cheaper per unit, pricier to build
    ax2.plot(volume, human, color=colors[0], linewidth=1.8, label="all-human labelling")
    ax2.plot(volume, engine, color=colors[2], linewidth=1.8, label="auto-label + verify the uncertain 12%")
    ax2.plot(volume, synth, color=colors[3], linewidth=1.8, label="synthetic + weak labels + spot checks")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("labels needed (log scale)")
    ax2.set_ylabel("cost per label, relative units (log scale)")
    ax2.set_title("Why data engines are a volume play", fontsize=10)
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.grid(alpha=0.25, linewidth=0.5)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.text(0.0, -0.20,
             "Schematic with a flat per-label price and a one-off engine build cost. The crossing point is the\n"
             "number you should estimate out loud in an interview, using your own price per label and build cost.",
             transform=ax2.transAxes, fontsize=7.5, color="#777777")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
