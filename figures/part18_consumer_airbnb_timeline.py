"""Timeline of Airbnb's published search-ranking model evolution.

Every entry is a published paper or Airbnb Tech Blog post (title in the label); the
year is the publication year, not the internal launch date.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_consumer_airbnb_timeline.png"

EVENTS = [
    (2018.0, "Listing embeddings\n(KDD 2018, real-time\npersonalization)", "C0", 1),
    (2018.8, "Applying Deep Learning to\nAirbnb Search (KDD 2019):\nGBDT -> simple NN -> LambdaRank\nNN -> deep NN", "C1", -1),
    (2020.0, "Improving Deep Learning for\nAirbnb Search (KDD 2020):\ncold start, position bias,\ncomplementary models", "C2", 1),
    (2020.5, "Managing Diversity in\nAirbnb Search (KDD 2020)", "C3", -1),
    (2022.3, "Learning To Rank Diversely\nat Airbnb (CIKM 2023)", "C4", 1),
    (2023.3, "Optimizing Airbnb Search\nJourney with Multi-task\nLearning (KDD 2023)", "C5", -1),
    (2024.3, "Chronon feature platform\nopen-sourced (Zipline lineage)", "C6", 1),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axhline(0, color="0.6", lw=1.5, zorder=1)
    for x, label, c, side in EVENTS:
        ax.plot([x], [0], marker="o", ms=9, color=c, zorder=3, markeredgecolor="white", markeredgewidth=1.5)
        ax.plot([x, x], [0, 0.55 * side], color=c, lw=1.2, zorder=2)
        ax.text(x, 0.6 * side, label, ha="center", va="bottom" if side > 0 else "top", fontsize=7.6, color="0.15")
    ax.set_xlim(2017.4, 2025.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_yticks([])
    ax.set_xticks(range(2018, 2025))
    ax.set_title("Airbnb search ranking: the published evolution (publication years)")
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
