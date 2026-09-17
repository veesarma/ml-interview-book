"""Position bias: observed click-through by rank confounds relevance with examination probability."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_position_bias.png"


def main() -> None:
    rng = np.random.default_rng(0)
    ranks = np.arange(1, 11)
    examination = 1.0 / np.sqrt(ranks)  # illustrative examination model P(E=1 | rank)
    true_relevance = np.full(10, 0.4) + rng.normal(0, 0.03, 10)  # items are roughly equally relevant
    observed_ctr = examination * true_relevance
    ips_recovered = observed_ctr / examination

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(ranks, observed_ctr, "o-", color="C0", lw=2, label="observed CTR = P(examine) x P(relevant)")
    ax1.plot(ranks, examination, "s--", color="C1", lw=2, label="P(examine | rank) (propensity)")
    ax1.set_xlabel("rank position")
    ax1.set_ylabel("probability")
    ax1.set_title("Clicks fall with rank even when relevance does not")
    ax1.legend(fontsize=8)
    ax2.plot(ranks, true_relevance, "o-", color="C2", lw=2, label="true relevance (unobserved)")
    ax2.plot(ranks, ips_recovered, "x", color="C3", ms=8, label="IPS-corrected: click / propensity")
    ax2.set_xlabel("rank position")
    ax2.set_ylabel("relevance")
    ax2.set_ylim(0, 0.8)
    ax2.set_title("Dividing by propensity recovers the flat truth")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        ax.set_xticks(ranks)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
