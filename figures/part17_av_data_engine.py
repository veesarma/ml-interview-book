"""AV perception: detection AP degrades with range, and the data-engine loop moves the tail.

All numbers are illustrative shapes for teaching, not any company's reported results.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_av_data_engine.png"


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: AP by range bucket for three classes -- the metric that a single mAP hides.
    ranges = ["0-30 m", "30-50 m", "50-80 m", "80-120 m", ">120 m"]
    x = np.arange(len(ranges))
    classes = {
        "vehicle": [0.94, 0.90, 0.82, 0.66, 0.41],
        "pedestrian": [0.88, 0.79, 0.61, 0.38, 0.17],
        "cyclist": [0.83, 0.72, 0.52, 0.29, 0.11],
    }
    width = 0.26
    for i, (name, aps) in enumerate(classes.items()):
        ax1.bar(x + (i - 1) * width, aps, width, label=name, color=f"C{i}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ranges)
    ax1.set_ylabel("average precision")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Report AP by class AND range, never one mAP")
    ax1.legend(fontsize=8)
    ax1.annotate("braking distance at 120 km/h\nstarts here", xy=(3.0, 0.38), xytext=(1.6, 0.20),
                 fontsize=8, arrowprops=dict(arrowstyle="->", lw=1))

    # Right: data-engine iterations vs failure rate on a mined long-tail scenario bucket.
    iters = np.arange(0, 9)
    random_collect = 0.20 * np.exp(-0.06 * iters)
    triggered = 0.20 * np.exp(-0.34 * iters)
    ax2.plot(iters, random_collect * 100, "o-", color="C0", lw=2, label="random fleet collection")
    ax2.plot(iters, triggered * 100, "s-", color="C2", lw=2, label="trigger-mined + auto-labelled + simulated")
    ax2.set_xlabel("data-engine iterations")
    ax2.set_ylabel("failure rate on the mined scenario bucket (%)")
    ax2.set_title("The data engine, not the architecture, moves the tail")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
