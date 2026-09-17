"""Open-vocabulary detection: a closed-set linear head vs region-text similarity.

Writes docs/assets/figures/part11_open_vocab.png
Run from the repository root:  python figures/part11_open_vocab.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_open_vocab.png"

REGIONS = ["region 1\n(sedan)", "region 2\n(cone)", "region 3\n(tipped\nscooter)", "region 4\n(road)"]
PHRASES = ["car", "traffic cone", "a scooter lying\non the road", "road surface", "police tape"]


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), facecolor="white",
                             gridspec_kw={"width_ratios": [1.0, 1.25]})

    # ---- left: the two head designs -------------------------------------------------
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9.5, "Where the class weights come from", ha="center", fontsize=10.5, weight="bold")

    ax.add_patch(plt.Rectangle((0.4, 5.6), 3.0, 2.4, facecolor=colors[0], alpha=0.18, edgecolor=colors[0]))
    ax.text(1.9, 7.6, "closed set", ha="center", fontsize=9.5, weight="bold")
    ax.text(1.9, 6.6, "r · W\nW is learned,\nK rows, frozen at\ntrain time", ha="center", fontsize=8.2)

    ax.add_patch(plt.Rectangle((5.6, 5.6), 4.0, 2.4, facecolor=colors[2], alpha=0.18, edgecolor=colors[2]))
    ax.text(7.6, 7.6, "open vocabulary", ha="center", fontsize=9.5, weight="bold")
    ax.text(7.6, 6.6, "r · t(phrase) / τ\nt comes from a text\nencoder: any phrase,\nno retraining", ha="center", fontsize=8.2)

    ax.annotate("", xy=(1.9, 5.4), xytext=(1.9, 4.6), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.annotate("", xy=(7.6, 5.4), xytext=(7.6, 4.6), arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.text(1.9, 4.2, "add a class\n= retrain the head", ha="center", fontsize=8.2, color="#b00020")
    ax.text(7.6, 4.2, "add a class\n= add a row of text", ha="center", fontsize=8.2, color="#1b7a3d")

    ax.text(5, 2.9, "cost of the flexibility", ha="center", fontsize=9.5, weight="bold")
    ax.text(5, 1.5, "text tower + fusion runs per frame or per vocabulary change;\n"
                    "a 300 MB grounded detector at 8 fps replaces a 20 MB\n"
                    "closed-set detector at 100 fps on the same chip.",
            ha="center", fontsize=8.2, color="#333333")

    # ---- right: the similarity matrix ------------------------------------------------
    ax = axes[1]
    rng = np.random.default_rng(0)
    sim = rng.uniform(0.02, 0.18, size=(len(REGIONS), len(PHRASES)))
    sim[0, 0] = 0.81           # sedan matches "car"
    sim[1, 1] = 0.76           # cone matches "traffic cone"
    sim[2, 2] = 0.63           # the long-tail object matches a full phrase
    sim[2, 0] = 0.24           # and weakly resembles a car
    sim[3, 3] = 0.72           # road
    im = ax.imshow(sim, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PHRASES)))
    ax.set_xticklabels(PHRASES, fontsize=8, rotation=18, ha="right")
    ax.set_yticks(range(len(REGIONS)))
    ax.set_yticklabels(REGIONS, fontsize=8)
    for i in range(len(REGIONS)):
        for j in range(len(PHRASES)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if sim[i, j] > 0.5 else "#444444")
    ax.set_title("cosine similarity  r_i · t_k  (the logits, before 1/τ and bias)", fontsize=9.5, loc="left")
    ax.set_xlabel("phrase embeddings from the text tower", fontsize=8.5)
    ax.set_ylabel("region features", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
