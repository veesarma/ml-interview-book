"""The StreetSmart fusion ablation: P95 latency against recall, shipped point marked.

Data verbatim from the reader's project deck: four ways of combining a frozen vision
embedding with a frozen text embedding, measured as P95 latency of the classification
stage in milliseconds and recall on the North American regulatory clusters.

Writes docs/assets/figures/part20_fusion_ablation.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part20_fusion_ablation.png"

# (variant, P95 latency ms, recall, shipped?)
ABLATION = [
    ("Concat", 12.4, 0.832, False),
    ("Learned scalar", 12.5, 0.844, False),
    ("Gated", 12.9, 0.857, False),
    ("Cross-attention", 14.2, 0.891, True),
]


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), facecolor="white")

    for name, lat, rec, shipped in ABLATION:
        if shipped:
            ax.scatter([lat], [rec], s=190, marker="*", color=colors[2],
                       zorder=3, label="shipped")
            ax.annotate(f"{name}\n{lat} ms, {rec:.3f}", (lat, rec),
                        textcoords="offset points", xytext=(-12, -34),
                        fontsize=8.5, color=colors[2], ha="center")
        else:
            ax.scatter([lat], [rec], s=70, color=colors[0], zorder=3)
            ax.annotate(f"{name}\n{lat} ms, {rec:.3f}", (lat, rec),
                        textcoords="offset points", xytext=(6, -2),
                        fontsize=8.5, color="#444444", va="center")

    lat_c, rec_c = ABLATION[0][1], ABLATION[0][2]
    lat_s, rec_s = ABLATION[3][1], ABLATION[3][2]
    ax.annotate("", xy=(lat_s, rec_s), xytext=(lat_c, rec_c),
                arrowprops=dict(arrowstyle="->", color="#aaaaaa", linewidth=1.2,
                                linestyle=(0, (3, 2))))
    ax.text((lat_c + lat_s) / 2 - 0.05, (rec_c + rec_s) / 2 + 0.006,
            f"+{100 * (rec_s - rec_c):.1f} pp recall\nfor +{lat_s - lat_c:.1f} ms P95",
            fontsize=8.5, color="#666666", ha="right")

    ax.set_xlabel("P95 latency of the classification stage (ms)")
    ax.set_ylabel("recall, NA regulatory clusters")
    ax.set_xlim(11.9, 15.6)
    ax.set_ylim(0.815, 0.905)
    ax.set_title("Fusion ablation: what the extra 1.8 ms bought", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.grid(color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
