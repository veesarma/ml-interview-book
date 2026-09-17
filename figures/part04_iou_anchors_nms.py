"""IoU geometry, an anchor grid with assignment, and NMS before/after."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from mlbook.detection.anchors import assign_max_iou, base_anchors, grid_anchors
from mlbook.detection.boxes import box_iou
from mlbook.detection.nms import nms

OUT = "docs/assets/figures/part04_iou_anchors_nms.png"


def draw_box(ax, b, **kw):
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, **kw))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="white")
    # IoU
    ax = axes[0]
    a = np.array([[1.0, 1.0, 6.0, 5.0]])
    b = np.array([[3.0, 2.0, 8.0, 7.0]])
    draw_box(ax, a[0], ec="C0", lw=2)
    draw_box(ax, b[0], ec="C1", lw=2)
    ax.add_patch(patches.Rectangle((3, 2), 3, 3, fc="C2", alpha=0.4))
    ax.text(4.5, 3.5, f"∩ = 9\nIoU = {box_iou(a, b)[0,0]:.3f}", ha="center", va="center")
    ax.set_xlim(0, 9)
    ax.set_ylim(8, 0)
    ax.set_aspect("equal")
    ax.set_title("IoU = |A∩B| / (|A|+|B|−|A∩B|)\n= 9 / (20 + 25 − 9)", fontsize=10)
    # anchors + assignment
    ax = axes[1]
    anchors = grid_anchors(3, 3, stride=20, templates=base_anchors(28.0, ratios=(0.5, 1.0, 2.0), scales=(1.0,)))
    gt = np.array([[16.0, 12.0, 48.0, 44.0]])
    labels, _ = assign_max_iou(anchors, gt, 0.5, 0.4)
    for anc, lab in zip(anchors, labels):
        color = {1: "C2", 0: "0.75", -1: "C1"}[int(lab)]
        draw_box(ax, anc, ec=color, lw=2.0 if lab == 1 else 1.0, alpha=1 if lab != 0 else 0.8)
    draw_box(ax, gt[0], ec="k", lw=2.5)
    ax.set_xlim(-12, 72)
    ax.set_ylim(72, -12)
    ax.set_aspect("equal")
    ax.set_title("3 ratios × 3×3 grid vs one GT (black)\ngreen = positive, orange = ignore, gray = negative", fontsize=10)
    # NMS
    ax = axes[2]
    rng = np.random.default_rng(1)
    centres = np.array([[20.0, 20.0], [55.0, 45.0]])
    boxes, scores = [], []
    for c in centres:
        for _ in range(8):
            jitter = rng.normal(0, 3, 2)
            wh = rng.uniform(18, 26, 2)
            boxes.append([c[0] + jitter[0] - wh[0] / 2, c[1] + jitter[1] - wh[1] / 2, c[0] + jitter[0] + wh[0] / 2, c[1] + jitter[1] + wh[1] / 2])
            scores.append(rng.uniform(0.3, 0.95))
    boxes, scores = np.array(boxes), np.array(scores)
    keep = nms(boxes, scores, 0.5)
    for i, b in enumerate(boxes):
        draw_box(ax, b, ec="C3" if i in keep else "0.7", lw=2.5 if i in keep else 0.8)
    for i in keep:
        ax.text(boxes[i, 0], boxes[i, 1] - 1, f"{scores[i]:.2f}", color="C3", fontsize=8)
    ax.set_xlim(0, 80)
    ax.set_ylim(80, 0)
    ax.set_aspect("equal")
    ax.set_title(f"NMS @ IoU 0.5\n{len(boxes)} boxes → {len(keep)} kept (red)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
