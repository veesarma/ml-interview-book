"""Gantt-style timeline of the 36-week study plan, one row per part of the book.

Writes docs/assets/figures/preface_timeline.png (dpi=150, tight bbox, white background).
Run from the repository root:  python figures/preface_timeline.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "preface_timeline.png"

# (row label, start week, end week inclusive, phase index). Weeks are 1-based.
# Phase index selects the colour from the default tab10 cycle (colourblind-safe subset used).
ROWS: list[tuple[str, int, int, int]] = [
    ("I. Math", 1, 3, 0),
    ("II. Classical ML", 4, 7, 0),
    ("III. Neural nets", 8, 10, 1),
    ("IV. Vision", 11, 13, 1),
    ("V. Sequences & Transformers", 14, 17, 2),
    ("VI. LLM training", 18, 20, 2),
    ("VIII. Multimodal", 21, 23, 3),
    ("X. Self-/semi-/weak supervision", 23, 23, 3),
    ("IX. Generative", 24, 25, 3),
    ("XIII. Retrieval, eval, reliability", 26, 26, 4),
    ("XII. Reinforcement learning", 27, 29, 5),
    ("VII. Post-training", 30, 32, 5),
    ("XI. Perception & autonomy", 33, 33, 6),
    ("XIV. Systems  /  XV. Safety", 34, 35, 6),
    ("XVI. Coding canon (weekly drills)", 4, 36, 7),
    ("XVII. System design (weekly mocks)", 20, 36, 7),
    ("XVIII. Company deep dives", 36, 36, 7),
    ("Capstone: multimodal reasoning model + post-training", 35, 36, 8),
]

PHASES = [
    "Foundations",
    "Nets & vision",
    "Transformers & LLMs",
    "Multimodal & generative",
    "Retrieval & eval",
    "RL & post-training",
    "Perception & systems",
    "Continuous practice",
    "Capstone",
]


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # default tab10
    fig, ax = plt.subplots(figsize=(11, 6.2), facecolor="white")

    n = len(ROWS)
    for i, (label, start, end, phase) in enumerate(ROWS):
        y = n - 1 - i
        width = end - start + 1
        hatch = "//" if phase == 7 else None  # continuous-practice rows are hatched
        ax.barh(
            y, width, left=start - 0.5, height=0.62,
            color=colors[phase], alpha=0.5 if phase == 7 else 0.9,
            edgecolor="white", linewidth=1.0, hatch=hatch,
        )
        ax.text(start - 0.5 + width / 2, y, f"W{start}" if start == end else f"W{start}–{end}",
                ha="center", va="center", fontsize=7.5, color="black")

    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in reversed(ROWS)], fontsize=8.5)
    ax.set_xlim(0.5, 36.5)
    ax.set_xticks(range(1, 37, 1))
    ax.set_xticklabels([str(w) if w % 2 == 1 else "" for w in range(1, 37)], fontsize=7.5)
    ax.set_xlabel("Week", fontsize=9)
    ax.set_title("The 36-week study plan, by part of the book", fontsize=11, loc="left")

    # Month markers (4.33 weeks per month) as light guides; grid is recessive.
    for m in range(1, 9):
        x = m * 4.33 + 0.5
        ax.axvline(x, color="#cccccc", linewidth=0.6, zorder=0)
        ax.text(x - 2.1, n - 0.35, f"month {m}", ha="center", va="bottom", fontsize=7, color="#666666")
    ax.set_ylim(-0.7, n + 0.3)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Patch(facecolor=colors[i], label=PHASES[i], alpha=0.5 if i == 7 else 0.9,
                     hatch="//" if i == 7 else None, edgecolor="white") for i in range(len(PHASES))]
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, ncol=3, frameon=False,
              bbox_to_anchor=(0.0, -0.26))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
