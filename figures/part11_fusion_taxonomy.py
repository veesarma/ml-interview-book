"""Early, intermediate and late fusion: where the sensors meet, and what each costs.

Writes docs/assets/figures/part11_fusion_taxonomy.png
Run from the repository root:  python figures/part11_fusion_taxonomy.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_fusion_taxonomy.png"


def stage(ax, x, y, w, h, text, color, alpha=0.85, fontsize=8.0):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=alpha, edgecolor="white", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="black")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.1))


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cam, lid, mix = colors[0], colors[1], colors[2]
    fig, axes = plt.subplots(3, 1, figsize=(11.6, 9.0), facecolor="white")

    rows = [
        ("Early fusion (point painting): raw sensor to raw sensor",
         "one detector, sharp geometry, cheapest head. Needs tight calibration and time sync;\n"
         "a camera failure poisons the input the LiDAR branch was trained on."),
        ("Intermediate fusion (TransFusion, BEVFusion): feature to feature",
         "each branch keeps its own encoder, a cross-attention or BEV concat mixes features.\n"
         "Best accuracy in the public benchmarks. Retraining is needed when either branch changes."),
        ("Late fusion (weighted box fusion, probabilistic): decision to decision",
         "independent detectors, independently testable, degrades to one sensor by construction.\n"
         "Loses everything that was only visible jointly: the faint radar return on the dark object."),
    ]

    for ax, (title, caption) in zip(axes, rows):
        ax.axis("off")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 22)
        ax.text(0, 20.4, title, fontsize=10.2, weight="bold")
        ax.text(0, 1.0, caption, fontsize=8.3, color="#333333")

    # --- early --------------------------------------------------------------------
    ax = axes[0]
    stage(ax, 2, 12, 13, 5, "camera\nimage", cam, 0.3)
    stage(ax, 2, 5, 13, 5, "LiDAR\npoints", lid, 0.3)
    stage(ax, 19, 12, 15, 5, "2D segmentation\nnetwork", cam)
    stage(ax, 38, 8.5, 15, 5, "paint points\nwith class scores", mix)
    stage(ax, 57, 8.5, 17, 5, "point-cloud\ndetector (x,y,z,p)", lid)
    stage(ax, 78, 8.5, 18, 5, "3D boxes", "#dddddd")
    arrow(ax, 15, 14.5, 19, 14.5); arrow(ax, 34, 14.5, 41, 13.5)
    arrow(ax, 15, 7.5, 41, 10.0); arrow(ax, 53, 11, 57, 11); arrow(ax, 74, 11, 78, 11)

    # --- intermediate --------------------------------------------------------------
    ax = axes[1]
    stage(ax, 2, 12, 13, 5, "camera\nimage", cam, 0.3)
    stage(ax, 2, 5, 13, 5, "LiDAR\npoints", lid, 0.3)
    stage(ax, 19, 12, 16, 5, "image backbone\n+ view transform", cam)
    stage(ax, 19, 5, 16, 5, "voxel / pillar\nbackbone", lid)
    stage(ax, 40, 8.5, 20, 5, "cross-attention\nin a shared BEV", mix)
    stage(ax, 64, 8.5, 14, 5, "BEV head", "#dddddd")
    stage(ax, 82, 8.5, 14, 5, "3D boxes\n+ map", "#dddddd")
    arrow(ax, 15, 14.5, 19, 14.5); arrow(ax, 15, 7.5, 19, 7.5)
    arrow(ax, 35, 14.5, 40, 12.5); arrow(ax, 35, 7.5, 40, 10.0)
    arrow(ax, 60, 11, 64, 11); arrow(ax, 78, 11, 82, 11)

    # --- late ----------------------------------------------------------------------
    ax = axes[2]
    stage(ax, 2, 12, 13, 5, "camera\nimage", cam, 0.3)
    stage(ax, 2, 5, 13, 5, "LiDAR\npoints", lid, 0.3)
    stage(ax, 19, 12, 17, 5, "camera detector", cam)
    stage(ax, 19, 5, 17, 5, "LiDAR detector", lid)
    stage(ax, 41, 12, 13, 5, "boxes A", "#dddddd")
    stage(ax, 41, 5, 13, 5, "boxes B", "#dddddd")
    stage(ax, 59, 8.5, 19, 5, "WBF / log-odds\nfusion", mix)
    stage(ax, 82, 8.5, 14, 5, "fused boxes", "#dddddd")
    arrow(ax, 15, 14.5, 19, 14.5); arrow(ax, 15, 7.5, 19, 7.5)
    arrow(ax, 36, 14.5, 41, 14.5); arrow(ax, 36, 7.5, 41, 7.5)
    arrow(ax, 54, 14.5, 59, 12.5); arrow(ax, 54, 7.5, 59, 10.0); arrow(ax, 78, 11, 82, 11)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
