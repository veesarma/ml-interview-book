"""The autonomous-vehicle data-engine flywheel as described publicly by Tesla
(AI Day 2021: triggers, auto-labelling, shadow mode) and Waymo (Waymax / SimulationCity
closed-loop evaluation). Eight stages on a ring, with the two evaluation loops that
gate deployment drawn as inner chords."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_av_data_engine_flywheel.png"

STAGES = [
    ("Fleet drives\n(customer / rider-only)", "C0"),
    ("On-vehicle triggers\n(disagreement, novelty,\nintervention)", "C1"),
    ("Clip upload +\nde-duplication", "C2"),
    ("Offline auto-labelling\n(multi-trip 4D reconstruction,\nbig offline models)", "C3"),
    ("Human QA on the\nhard / ambiguous slice", "C4"),
    ("Train (multi-task /\nend-to-end)", "C5"),
    ("Evaluate: replay, closed-loop\nsim, shadow mode", "C6"),
    ("Ship OTA to\nthe fleet", "C7"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")

    n = len(STAGES)
    radius = 1.0
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n  # (n,) clockwise from 12 o'clock
    xs, ys = radius * np.cos(angles), radius * np.sin(angles)

    for i, ((label, color), x, y) in enumerate(zip(STAGES, xs, ys)):
        box = FancyBboxPatch((x - 0.30, y - 0.14), 0.60, 0.28, boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor=color, alpha=0.18, edgecolor=color, lw=1.6)
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=7.6, color="0.15")
        nxt = (i + 1) % n
        # shorten the arrow so it starts / ends outside the boxes
        dx, dy = xs[nxt] - x, ys[nxt] - y
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        arrow = FancyArrowPatch((x + 0.34 * ux, y + 0.19 * uy), (xs[nxt] - 0.34 * ux, ys[nxt] - 0.19 * uy),
                                arrowstyle="-|>", mutation_scale=14, color="0.35", lw=1.2,
                                connectionstyle="arc3,rad=-0.25")
        ax.add_patch(arrow)

    # Inner chords: the two evaluation gates
    i_eval, i_trig, i_train = 6, 1, 5
    ax.add_patch(FancyArrowPatch((xs[i_eval] + 0.2, ys[i_eval] - 0.05), (xs[i_trig] - 0.25, ys[i_trig] - 0.1),
                                 arrowstyle="-|>", mutation_scale=12, color="C1", lw=1.2, linestyle="--",
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(0.05, 0.30, "failed cases become\nnew triggers", ha="center", fontsize=7.5, color="C1")
    ax.add_patch(FancyArrowPatch((xs[i_eval] - 0.05, ys[i_eval] - 0.17), (xs[i_train] - 0.28, ys[i_train] + 0.10),
                                 arrowstyle="-|>", mutation_scale=12, color="C6", lw=1.2, linestyle="--",
                                 connectionstyle="arc3,rad=0.3"))
    ax.text(-1.22, -0.50, "regression on a\nscenario slice:\nback to training", ha="center", fontsize=7.5, color="C6")

    ax.text(0, -1.38, "Flywheel stages as described in Tesla AI Day 2021 / CVPR'21 WAD keynote (triggers, auto-labelling, "
            "shadow mode)\nand Waymo's simulation posts (SimulationCity 2021, Waymax 2023). Loop speed, not any single "
            "model, is the moat.", ha="center", fontsize=7.6, color="0.35")
    ax.set_title("The AV data engine: a flywheel gated by evaluation", fontsize=12, pad=6)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
