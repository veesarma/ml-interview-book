"""Multi-stage recommendation funnel with the stage sizes YouTube published.

Covington, Adams & Sargin, "Deep Neural Networks for YouTube Recommendations"
(RecSys 2016) describe a two-stage system: candidate generation reduces a corpus of
"millions" of videos to "hundreds", and ranking selects the "dozens" that are shown.
Only orders of magnitude are public, so the bars below are drawn at 1e6 / 1e2 / 1e1
and labelled with the paper's own words.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_consumer_rec_funnel.png"


def main() -> None:
    stages = [
        ("Corpus", 1e6, '"millions" of videos'),
        ("Candidate generation\n(two-tower / ANN)", 1e2, '"hundreds" of candidates'),
        ("Ranking\n(heavy multi-task model)", 1e1, '"dozens" shown to the user'),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(stages))[::-1]
    widths = [np.log10(s[1]) for s in stages]
    ax.barh(y, widths, height=0.55, color=["C0", "C1", "C2"], edgecolor="white", linewidth=2)
    for yi, (name, size, words), w in zip(y, stages, widths):
        ax.text(-0.15, yi, name, ha="right", va="center", fontsize=9)
        ax.text(w + 0.1, yi, words, ha="left", va="center", fontsize=9, color="0.25")
    ax.set_xlim(0, 8.5)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([f"$10^{k}$" for k in range(7)])
    ax.set_xlabel("items surviving the stage (log scale)")
    ax.set_title("The two-stage funnel, with the orders of magnitude YouTube published (RecSys 2016)")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
