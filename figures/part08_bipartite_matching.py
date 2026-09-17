"""DETR bipartite matching: Q predictions vs T ground-truth boxes, the cost matrix, and the Hungarian assignment.

Writes docs/assets/figures/part08_bipartite_matching.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlbook.multimodal.hungarian import hungarian  # noqa: E402

OUT = ROOT / "docs" / "assets" / "figures" / "part08_bipartite_matching.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    gt = [(0.15, 0.2, 0.3, 0.35), (0.6, 0.5, 0.3, 0.4)]  # x, y, w, h
    preds = [(0.17, 0.22, 0.28, 0.33), (0.62, 0.55, 0.3, 0.35), (0.4, 0.1, 0.2, 0.2), (0.1, 0.7, 0.25, 0.2)]
    C = np.array([[0.3, 2.8], [2.6, 0.4], [1.9, 2.1], [2.4, 2.9]])
    rows, cols = hungarian(C)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), facecolor="white", gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    for k, (x, y, w, h) in enumerate(gt):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="black", linewidth=2.5, linestyle="-"))
        ax.text(x, y + h + 0.02, f"GT {k}", fontsize=8)
    for q, (x, y, w, h) in enumerate(preds):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=colors[q], linewidth=1.8, linestyle="--"))
        ax.text(x + w, y, f"q{q}", fontsize=8, color=colors[q], ha="right", va="top")
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal")
    ax.set_title("Q = 4 queries (dashed) vs T = 2 targets (solid)", fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow(C, cmap="Blues")
    for i in range(4):
        for j in range(2):
            ax.text(j, i, f"{C[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if (i, j) in zip(rows, cols) else "normal")
    for i, j in zip(rows, cols):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GT 0", "GT 1"], fontsize=8)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"q{i}" for i in range(4)], fontsize=8)
    ax.set_title("cost C[q,t] = -p(c_t) + 5·L1 - 2·GIoU", fontsize=9, loc="left")

    ax = axes[2]
    for q in range(4):
        ax.scatter(0, 3 - q, s=250, color=colors[q], zorder=3)
        ax.text(-0.15, 3 - q, f"q{q}", ha="right", va="center", fontsize=9)
    for t in range(2):
        ax.scatter(1, 2.5 - t, s=250, color="black", zorder=3)
        ax.text(1.15, 2.5 - t, f"GT {t}", ha="left", va="center", fontsize=9)
    ax.scatter(1, 0.5, s=250, facecolor="white", edgecolor="grey", zorder=3)
    ax.text(1.15, 0.5, "∅ (no object)", ha="left", va="center", fontsize=9, color="grey")
    for i, j in zip(rows, cols):
        ax.plot([0, 1], [3 - i, 2.5 - j], color=colors[i], linewidth=2.5)
    matched = set(rows.tolist())
    for q in range(4):
        if q not in matched:
            ax.plot([0, 1], [3 - q, 0.5], color=colors[q], linewidth=1.5, linestyle=":")
    ax.set_xlim(-0.6, 1.9); ax.set_ylim(-0.2, 3.5); ax.axis("off")
    ax.set_title("Hungarian assignment: unmatched queries → ∅", fontsize=9, loc="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
