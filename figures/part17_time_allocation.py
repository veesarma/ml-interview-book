"""Time allocation for a 45- and 60-minute ML system design round (Part XVII framework)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_time_allocation.png"

PHASES = [
    ("Clarify & scope", 5, 7),
    ("Metrics & success", 4, 5),
    ("Data & labels", 6, 8),
    ("Model & funnel", 12, 16),
    ("Training & serving", 8, 11),
    ("Evaluation & A/B", 5, 7),
    ("Deep dives / wrap-up", 5, 6),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for row, (label, total) in enumerate([("45 min", 45), ("60 min", 60)]):
        left = 0.0
        for i, (name, m45, m60) in enumerate(PHASES):
            width = m45 if total == 45 else m60
            ax.barh(row, width, left=left, color=f"C{i}", edgecolor="white", linewidth=2, height=0.55,
                    label=name if row == 0 else None)
            if width >= 4:
                ax.text(left + width / 2, row, f"{width}", ha="center", va="center", color="white", fontsize=9)
            left += width
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["45-minute round", "60-minute round"])
    ax.set_xlabel("minutes")
    ax.set_xlim(0, 62)
    ax.set_title("Where the minutes go: the model phase gets a third, never more than half")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
