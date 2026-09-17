"""The (B, T, d_model) -> (B, T, H, d_head) -> (B, H, T, d_head) walk-through as a diagram.
Writes docs/assets/figures/part05_heads_reshape.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_heads_reshape.png"
T, H, DH = 4, 3, 2  # tiny example: d_model = 6


def draw_matrix(ax, x0, y0, rows, cols, cell, colors, labels=None, lw=1.0):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((x0 + c * cell, y0 - (r + 1) * cell), cell, cell, facecolor=colors[r][c], edgecolor="black", linewidth=lw))
            if labels is not None:
                ax.text(x0 + (c + 0.5) * cell, y0 - (r + 0.5) * cell, labels[r][c], ha="center", va="center", fontsize=6.5)


def main() -> None:
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    head_color = [palette[h] for h in range(H)]
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6)
    ax.axis("off")
    cell = 0.55
    # 1) (T, d_model): columns coloured by the head they will belong to
    colors = [[head_color[c // DH] for c in range(H * DH)] for _ in range(T)]
    labels = [[f"t{r}" for _ in range(H * DH)] for r in range(T)]
    draw_matrix(ax, 0.5, 5, T, H * DH, cell, colors, labels)
    ax.text(0.5, 5.25, "x = W_q(X)   (B, T, d_model)", fontsize=9)
    ax.text(0.5, 1.9, "one row per token; d_model = H * d_head\ncolumns are already grouped by head", fontsize=7.5, va="top")
    # 2) view -> (T, H, d_head): same memory, drawn as H side-by-side blocks per row
    x1 = 6.5
    for h in range(H):
        colors = [[head_color[h]] * DH for _ in range(T)]
        labels = [[f"t{r}"] * DH for r in range(T)]
        draw_matrix(ax, x1 + h * (DH * cell + 0.25), 5, T, DH, cell, colors, labels)
        ax.text(x1 + h * (DH * cell + 0.25) + DH * cell / 2, 5.05, f"h{h}", ha="center", fontsize=7)
    ax.text(x1, 5.5, ".view(B, T, H, d_head)   [free: no copy]", fontsize=9)
    ax.text(x1, 1.9, "split the last axis into (H, d_head)\nstill one row per token", fontsize=7.5, va="top")
    # 3) transpose -> (H, T, d_head): one (T, d_head) matrix per head, stacked
    x2 = 13.0
    for h in range(H):
        colors = [[head_color[h]] * DH for _ in range(T)]
        labels = [[f"t{r}"] * DH for r in range(T)]
        draw_matrix(ax, x2 + h * (DH * cell + 0.9), 5, T, DH, cell, colors, labels)
        ax.text(x2 + h * (DH * cell + 0.9) + DH * cell / 2, 5.05, f"head {h}: (T, d_head)", ha="center", fontsize=7)
    ax.text(x2, 5.5, ".transpose(1, 2)   -> (B, H, T, d_head)", fontsize=9)
    ax.text(x2, 1.9, "H becomes a batch axis: q @ k.transpose(-2, -1)\nruns H independent (T x T) attentions", fontsize=7.5, va="top")
    for xa, xb in ((4.0, 6.3), (10.9, 12.8)):
        ax.add_patch(FancyArrowPatch((xa, 3.9), (xb, 3.9), arrowstyle="->", mutation_scale=14, linewidth=1.2))
    ax.text(11, 0.5, "merge_heads reverses it: transpose(1, 2) -> contiguous() -> view(B, T, H * d_head), then W_o", ha="center", fontsize=8.5, style="italic")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
