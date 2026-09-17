"""Per-cluster recall on North American sign clusters, before and after OCR fusion.

Data is taken verbatim from the reader's StreetSmart project deck (Q4 2024 - Q2 2025):
recall in percent for each sign cluster, vision-only ("NA pre") versus the frozen
ViT + frozen BERT cross-attention fusion classifier ("NA post"), with the European
vision-only baseline shown as a reference marker where a European analog exists.

Writes docs/assets/figures/part20_cluster_recall.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part20_cluster_recall.png"

# (cluster, EU recall or None, NA pre recall, NA post recall)
CLUSTERS = [
    ("Stop / yield", 97.1, 93.2, 97.4),
    ("Speed limit", 96.0, 71.4, 95.8),
    ("Turn restrictions", 94.5, 68.9, 94.1),
    ("Truck restrictions", 93.4, 62.7, 91.2),
    ("Time-qualified parking", None, 58.2, 88.6),
    ("Construction / work zone", 92.1, 66.5, 90.8),
    ("Guide / informational", 89.2, 70.4, 88.1),
]

CUSTOMER_FLOOR = 96.0


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    names = [c[0] for c in CLUSTERS]
    eu = [c[1] for c in CLUSTERS]
    pre = np.array([c[2] for c in CLUSTERS])
    post = np.array([c[3] for c in CLUSTERS])

    x = np.arange(len(CLUSTERS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor="white")
    ax.bar(x - width / 2, pre, width, color=colors[1], label="NA, vision only (pre)")
    ax.bar(x + width / 2, post, width, color=colors[0], label="NA, OCR fusion (post)")

    for i, (a, b) in enumerate(zip(pre, post)):
        ax.annotate(f"{a:.1f}", (i - width / 2, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color="#444444")
        ax.annotate(f"{b:.1f}", (i + width / 2, b), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7.5, color="#444444")
        ax.annotate(f"+{b - a:.1f} pp", (i, max(a, b) + 2.6), ha="center",
                    fontsize=8.5, color="#333333", fontweight="bold")

    for i, v in enumerate(eu):
        if v is None:
            continue
        ax.plot([i - 1.7 * width, i + 0.05 * width], [v, v], linestyle=(0, (4, 2)),
                linewidth=1.4, color=colors[7])
    ax.plot([], [], linestyle=(0, (4, 2)), linewidth=1.4, color=colors[7],
            label="EU baseline, vision only")

    ax.axhline(CUSTOMER_FLOOR, color="#999999", linewidth=0.9, linestyle="-.")
    ax.text(4.0, CUSTOMER_FLOOR + 1.0, "contracted recall floor, 96%",
            fontsize=7.5, color="#666666", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
    ax.set_ylabel("recall (%)")
    ax.set_ylim(50, 104)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_title("StreetSmart: per-cluster recall, vision only vs OCR fusion (read by row)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=1)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
