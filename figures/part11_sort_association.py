"""SORT: the IoU cost matrix, the Hungarian assignment, and what ByteTrack's second pass adds.

Writes docs/assets/figures/part11_sort_association.png
Run from the repository root:  python figures/part11_sort_association.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.perception.hungarian import hungarian  # noqa: E402
from mlbook.perception.sort_tracker import iou_matrix  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_sort_association.png"

PRED = np.array([[10.0, 10.0, 50.0, 70.0],
                 [70.0, 20.0, 108.0, 78.0],
                 [130.0, 15.0, 168.0, 73.0]])
DETS = np.array([[14.0, 12.0, 54.0, 72.0],
                 [66.0, 24.0, 104.0, 82.0],
                 [190.0, 30.0, 226.0, 86.0]])
DET_SCORE = [0.93, 0.28, 0.88]


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    iou = iou_matrix(PRED, DETS)
    cost = 1.0 - iou
    rows, cols = hungarian(cost)
    accepted = cost[rows, cols] <= 0.7  # iou_threshold = 0.3

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), facecolor="white")

    # ---- 1. boxes in the image plane --------------------------------------------------
    ax = axes[0]
    for i, b in enumerate(PRED):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   edgecolor=colors[0], lw=1.8, ls="--"))
        ax.text(b[0], b[1] - 4, f"track {i+1} (KF prediction)", fontsize=7.5, color=colors[0])
    for j, b in enumerate(DETS):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                   edgecolor=colors[3], lw=1.8))
        ax.text(b[0], b[3] + 9, f"det {j+1}  s={DET_SCORE[j]:.2f}", fontsize=7.5, color=colors[3])
    ax.set_xlim(0, 245)
    ax.set_ylim(110, -12)
    ax.set_aspect("equal")
    ax.set_title("predictions (dashed) and detections (solid)", fontsize=9.8, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    # ---- 2. the cost matrix and the assignment ---------------------------------------
    ax = axes[1]
    im = ax.imshow(cost, cmap="RdYlGn_r", vmin=0.0, vmax=1.0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cost[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="black")
    for r, c, ok in zip(rows, cols, accepted):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor="black" if ok else "#888888",
                                   lw=2.6, ls="-" if ok else ":"))
    ax.set_xticks(range(3), [f"det {j+1}" for j in range(3)], fontsize=8.5)
    ax.set_yticks(range(3), [f"track {i+1}" for i in range(3)], fontsize=8.5)
    ax.set_title(r"cost $= 1 - \mathrm{IoU}$, boxed = Hungarian pick", fontsize=9.8, loc="left")
    ax.text(-0.45, 2.85, "dotted box: chosen by the assignment but rejected by the\n"
                         "IoU gate (cost > 0.7), so track 3 coasts and det 3 is born",
            fontsize=7.8, color="#333333")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    # ---- 3. the two-pass ByteTrack idea ----------------------------------------------
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.5, "ByteTrack: a second pass on the leftovers", fontsize=9.8, weight="bold")
    ax.add_patch(plt.Rectangle((0.3, 6.6), 9.2, 2.2, facecolor=colors[2], alpha=0.16, edgecolor=colors[2]))
    ax.text(0.6, 8.2, "pass 1:  tracks  <->  detections with s >= 0.6", fontsize=8.6)
    ax.text(0.6, 7.2, "det 1 (0.93) and det 3 (0.88) are considered here.\ndet 2 (0.28) is not.", fontsize=8.0)
    ax.add_patch(plt.Rectangle((0.3, 3.4), 9.2, 2.6, facecolor=colors[1], alpha=0.16, edgecolor=colors[1]))
    ax.text(0.6, 5.4, "pass 2:  still-unmatched tracks  <->  0.1 <= s < 0.6", fontsize=8.6)
    ax.text(0.6, 4.1, "track 2 now matches det 2 and keeps its identity.\nPlain SORT would have dropped det 2, coasted,\nthen spawned a new id when the score recovered.", fontsize=8.0)
    ax.text(0.3, 2.2, "Only high-score leftovers may start a new track: a low-score\n"
                      "box is trusted to continue an identity, never to create one.",
            fontsize=8.0, color="#333333")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
