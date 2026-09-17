"""Novel-object 3D detection: why static objects are easy offline, and the tri-modal agreement funnel.

Shapes are illustrative for teaching, not measurements from any dataset.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_static_aggregation.png"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    # Left: point density on a static object grows with passes; a dynamic object does not.
    passes = np.arange(1, 41)
    static_single = 120 * passes                      # every pass adds points in the world frame
    static_sweeps = 120 * passes * 1.0
    dynamic = 120 * np.ones_like(passes) * 1.6        # per-timestep, object frame, bounded by one pass
    ax1.plot(passes, static_sweeps, lw=2, color="C0", label="static object: aggregate in world frame")
    ax1.plot(passes, dynamic, lw=2, color="C3", label="dynamic object: per-timestep, object frame")
    ax1.set_yscale("log")
    ax1.set_xlabel("fleet passes over the same location")
    ax1.set_ylabel("LiDAR points available on the object (log)")
    ax1.set_title("Never moving is a gift: aggregation is unbounded")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.annotate("human-labellable\nfrom one pass", xy=(1, 120), xytext=(6, 200),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=1))

    # Right: the tri-modal weak-supervision funnel for a zero-label class.
    stages = ["candidates\nproposed", "open-vocab 2D\nlifted to 3D", "+ intensity\nagreement", "+ map / topology\nprior", "auto-accepted\npseudo-labels"]
    counts = [100, 62, 41, 34, 31]
    audit = [0, 0, 0, 0, 9]
    x = np.arange(len(stages))
    ax2.bar(x, counts, 0.6, color="C0", edgecolor="white", linewidth=2, label="surviving candidates (%)")
    ax2.bar(x, audit, 0.6, bottom=counts, color="C1", edgecolor="white", linewidth=2,
            label="routed to ranked human audit (%)")
    for xi, c in zip(x, counts):
        ax2.text(xi, c + 2, f"{c}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=8)
    ax2.set_ylabel("share of initial candidates (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Agreement between weak signals buys precision without labels")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
