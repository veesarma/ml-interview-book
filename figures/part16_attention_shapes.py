"""The multi-head attention shape pipeline, and why the merge needs contiguity.

Writes docs/assets/figures/part16_attention_shapes.png.
Run from the repository root: python figures/part16_attention_shapes.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part16_attention_shapes.png"

BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#666666"

# (label, shape, x, colour)
STAGES = [
    ("x", "(B, T, d)", 0.6, BLUE),
    ("q = q_proj(x)", "(B, T, d)", 2.3, BLUE),
    (".reshape", "(B, T, H, dh)", 4.0, ORANGE),
    (".transpose(1, 2)", "(B, H, T, dh)", 5.9, ORANGE),
    ("q @ k.transpose(-2,-1)", "(B, H, T, S)", 8.0, RED),
    ("softmax @ v", "(B, H, T, dh)", 10.2, GREEN),
    (".transpose(1, 2)", "(B, T, H, dh)", 12.1, ORANGE),
    (".reshape", "(B, T, d)", 13.9, BLUE),
]

BOX_W, BOX_H = 1.5, 0.9


def pipeline(ax) -> None:
    for i, (label, shape, x, colour) in enumerate(STAGES):
        ax.add_patch(FancyBboxPatch(
            (x - BOX_W / 2, 1.0 - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            linewidth=1.6, edgecolor=colour, facecolor="white", zorder=2,
        ))
        ax.text(x, 1.20, label, ha="center", va="center", fontsize=7.6, zorder=3)
        ax.text(x, 0.86, shape, ha="center", va="center", fontsize=8.4,
                family="monospace", color=colour, fontweight="bold", zorder=3)
        if i:
            prev = STAGES[i - 1][2]
            ax.add_patch(FancyArrowPatch(
                (prev + BOX_W / 2, 1.0), (x - BOX_W / 2, 1.0),
                arrowstyle="-|>", mutation_scale=10, linewidth=1.1, color=GREY, zorder=1,
            ))

    notes = [
        (4.95, "d = H * dh,\nno data moves"),
        (6.95, "heads become\na batch axis"),
        (9.1, "mask added here;\nS = T, or T_kv with a cache"),
        (11.15, "weights @ v\nputs T back"),
        (13.0, "merge needs\n.contiguous()"),
    ]
    for x, text in notes:
        ax.text(x, 0.30, text, ha="center", va="top", fontsize=7.2, color="#444444")

    ax.set_xlim(-0.3, 14.9)
    ax.set_ylim(-0.42, 1.62)
    ax.axis("off")
    ax.set_title("One head split, one matmul, one merge", fontsize=12, pad=6)


def memory_panel(ax) -> None:
    """B=1, T=2, H=2, dh=2 laid out as memory indices, before and after the transpose."""
    cell = 0.62

    def grid(x0, y0, rows, title, subtitle, colour):
        ax.text(x0 + cell * 2, y0 + cell * len(rows) + 0.30, title,
                ha="center", fontsize=9.2, fontweight="bold")
        ax.text(x0 + cell * 2, y0 + cell * len(rows) + 0.06, subtitle,
                ha="center", fontsize=8.0, family="monospace", color=colour)
        for r, (row_label, values) in enumerate(rows):
            y = y0 + cell * (len(rows) - 1 - r)
            ax.text(x0 - 0.16, y + cell / 2, row_label, ha="right", va="center", fontsize=7.6)
            for c, v in enumerate(values):
                contiguous = c == 0 or v == values[c - 1] + 1
                ax.add_patch(Rectangle(
                    (x0 + c * cell, y), cell, cell,
                    facecolor="#f2f2f2" if contiguous else "#ffe9e9",
                    edgecolor=colour, linewidth=1.2,
                ))
                ax.text(x0 + c * cell + cell / 2, y + cell / 2, str(v),
                        ha="center", va="center", fontsize=8.6, family="monospace")

    before = [
        ("t=0", [0, 1, 2, 3]),
        ("t=1", [4, 5, 6, 7]),
    ]
    after = [
        ("h=0", [0, 1, 4, 5]),
        ("h=1", [2, 3, 6, 7]),
    ]
    grid(0.7, 0.4, before, "after .reshape(B, T, H, dh)", "row = one token, 4 values", ORANGE)
    grid(4.6, 0.4, after, "after .transpose(1, 2)", "row = one head, 4 values", RED)

    ax.add_patch(FancyArrowPatch((3.5, 1.02), (4.35, 1.02), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.3, color=GREY))
    ax.text(3.93, 1.14, ".transpose", ha="center", fontsize=7.6, color=GREY)

    ax.text(7.5, 1.92,
            "The numbers are positions in the underlying buffer, which the transpose never touches.",
            fontsize=8.6, ha="left", va="center")
    ax.text(7.5, 1.60,
            "Reading the right-hand grid row by row needs 0,1,4,5: not consecutive, so the view is\n"
            "no longer contiguous. .reshape copies when it has to; .view raises instead. Calling\n"
            ".contiguous() before .view is what makes the merge legal, and it costs one copy of\n"
            "the activations.",
            fontsize=8.2, ha="left", va="top")
    ax.text(7.5, 0.55,
            "Skipping the transpose on the way back, so merging (B, H, T, dh) straight to (B, T, d),\n"
            "silently interleaves the heads. The shapes still check out and the loss still falls.",
            fontsize=8.2, ha="left", va="top", color=RED)

    ax.set_xlim(-0.2, 15.0)
    ax.set_ylim(0.0, 2.25)
    ax.axis("off")


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 5.9), dpi=150,
                             gridspec_kw={"height_ratios": [1.0, 1.15]})
    fig.patch.set_facecolor("white")
    pipeline(axes[0])
    memory_panel(axes[1])
    fig.tight_layout(h_pad=0.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
