"""Raw precision/recall points from a toy detector, the monotone envelope used by
all-points AP, and the 11 VOC-2007 sample points.
Writes docs/assets/figures/part13_ap_interpolation.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.evaluation.detection_map import ap_all_points, ap_voc07, precision_recall_from_matches  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_ap_interpolation.png"


def main() -> None:
    tp = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=float)  # (D,) score-sorted
    fp = 1 - tp
    n_gt = 8
    p, r = precision_recall_from_matches(tp, fp, n_gt)
    env = p.copy()
    for i in range(len(env) - 2, -1, -1):
        env[i] = max(env[i], env[i + 1])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.step(np.r_[0, r], np.r_[1, p], where="post", color="C0", label="raw precision (score-sorted)")
    ax.plot(r, p, "o", color="C0", ms=4)
    ax.step(np.r_[0, r, 1], np.r_[env[0], env, 0], where="post", color="C3", lw=2, label="monotone envelope (all-points AP)")
    for rr in np.linspace(0, 1, 11):
        pm = p[r >= rr].max() if np.any(r >= rr) else 0.0
        ax.plot(rr, pm, "s", color="C2", ms=7, mfc="none", mew=1.8)
    ax.plot([], [], "s", color="C2", mfc="none", mew=1.8, label="11-point samples (VOC07)")
    ax.set(xlabel="recall", ylabel="precision", xlim=(-0.02, 1.02), ylim=(0, 1.05),
           title=f"AP(all-points)={ap_all_points(p, r):.3f}   AP(11-pt)={ap_voc07(p, r):.3f}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
