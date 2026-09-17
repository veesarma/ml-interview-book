"""CARL time budget for a 45-minute and a 60-minute presented project deep dive.

The bars show planned speaking minutes per section. The unallocated tail is the
interruption reserve: in a real deep dive the interviewer takes 30-40% of the slot
with questions, so a deck planned to fill the whole slot never reaches its Learnings.

Writes docs/assets/figures/part20_carl_time_budget.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part20_carl_time_budget.png"

SECTIONS = [
    "TL;DR",
    "C: context",
    "C: problem",
    "C: constraints",
    "A: options + architecture",
    "R: results",
    "R: regressions",
    "L: learnings + next",
    "interruption reserve",
]

MIN45 = [2, 4, 5, 3, 9, 3, 2, 2, 15]
MIN60 = [2, 5, 6, 4, 13, 4, 3, 3, 20]


def main() -> None:
    cmap = plt.get_cmap("tab20c")
    colors = [cmap(i / 20) for i in (0, 4, 5, 6, 8, 12, 13, 16, 19)]

    fig, ax = plt.subplots(figsize=(10.0, 3.4), facecolor="white")
    rows = [("60-minute slot", MIN60, 1.0), ("45-minute slot", MIN45, 0.0)]

    for label, mins, y in rows:
        left = 0.0
        for i, m in enumerate(mins):
            ax.barh(y, m, left=left, height=0.52, color=colors[i],
                    edgecolor="white", linewidth=1.0)
            if m >= 3:
                ax.text(left + m / 2, y, f"{m}", ha="center", va="center",
                        fontsize=8.5, color="#222222")
            left += m
        ax.text(-1.0, y, label, ha="right", va="center", fontsize=9.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(SECTIONS))]
    ax.legend(handles, SECTIONS, fontsize=8, frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))

    ax.set_xlim(0, 61)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, 61, 5))
    ax.set_xlabel("minutes")
    ax.set_title("CARL time budget: what you plan to say, and what you leave for them",
                 fontsize=10.5, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
