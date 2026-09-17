"""Broadcasting diagram: trailing-dimension alignment, size-1 stretching, and the
attention-mask case (B, 1, 1, T) against (B, H, T, T)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_broadcasting.png"


def draw_grid(ax, x0, y0, rows, cols, color, label, stretched_rows=False, stretched_cols=False, cell=0.42):
    for r in range(rows):
        for c in range(cols):
            hatch = "//" if (stretched_rows and r > 0) or (stretched_cols and c > 0) else None
            ax.add_patch(
                Rectangle((x0 + c * cell, y0 - r * cell), cell, cell, facecolor=color, edgecolor="0.3", alpha=0.35 if hatch else 0.8, hatch=hatch, lw=0.8)
            )
    ax.text(x0 + cols * cell / 2, y0 + 0.55, label, ha="center", fontsize=9)


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(-2.4, 2.2)
    ax.axis("off")

    # Example 1: (3, 4) + (4,) -> (3, 4)
    draw_grid(ax, 0.3, 1.0, 3, 4, "C0", "A  (3, 4)")
    ax.text(2.4, 0.45, "+", fontsize=16, ha="center")
    draw_grid(ax, 2.7, 1.0, 1, 4, "C1", "b  (4,) → (1, 4)")
    ax.text(4.85, 0.45, "=", fontsize=16, ha="center")
    draw_grid(ax, 5.1, 1.0, 3, 4, "C1", "b stretched to (3, 4)", stretched_rows=True)
    ax.text(7.75, -1.0, "Rule: align shapes from the right; a missing leading dim counts as 1; any size-1 dim is stretched (a stride-0 view, no copy); anything else is an error.", fontsize=8.5, ha="center", color="0.25")

    # Example 2: (3, 1) * (1, 4) -> (3, 4) outer product
    draw_grid(ax, 8.2, 1.0, 3, 1, "C2", "u  (3, 1)")
    ax.text(9.0, 0.45, "×", fontsize=16, ha="center")
    draw_grid(ax, 9.5, 1.0, 1, 4, "C3", "v  (1, 4)")
    ax.text(11.6, 0.45, "=", fontsize=16, ha="center")
    draw_grid(ax, 11.9, 1.0, 3, 4, "C4", "u vᵀ  (3, 4): outer product")
    ax.text(7.75, -1.5, "Left: (3,4) + (4,) → b becomes (1,4) then stretches down rows.   Right: (3,1) × (1,4) → both stretch: the outer product.", fontsize=8.5, ha="center", color="0.25")
    ax.text(7.75, -2.0, "Same rule at scale: causal mask (1,1,T,T) or padding mask (B,1,1,T) added to scores (B,H,T,T).", fontsize=8.5, ha="center", color="0.25")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
