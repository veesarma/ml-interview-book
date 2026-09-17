"""Content moderation: review capacity fixes the operating point; prevalence is what the user experiences."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_moderation_prevalence.png"


def main() -> None:
    rng = np.random.default_rng(7)
    n = 1_000_000
    base_rate = 0.005  # 0.5% of uploads violate
    y = rng.random(n) < base_rate
    score = np.where(y, rng.normal(1.8, 1.0, n), rng.normal(0.0, 1.0, n))
    thresholds = np.linspace(-1, 5, 200)
    flagged = np.array([(score > t).sum() for t in thresholds])
    caught = np.array([(y & (score > t)).sum() for t in thresholds])
    precision = caught / np.maximum(flagged, 1)
    recall = caught / y.sum()
    prevalence_after = (y.sum() - caught) / n  # violating content that reaches users, if flagged is removed

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(thresholds, flagged / n * 100, color="C0", lw=2, label="% of uploads flagged (review load)")
    ax1.plot(thresholds, precision * 100, color="C1", lw=2, label="precision of flags (%)")
    ax1.plot(thresholds, recall * 100, color="C2", lw=2, label="recall (%)")
    ax1.axhline(1.0, color="k", ls="--", lw=1)
    ax1.text(2.5, 1.4, "human review capacity = 1% of uploads", fontsize=8)
    ax1.set_xlabel("score threshold")
    ax1.set_ylabel("percent")
    ax1.set_ylim(0, 100)
    ax1.set_title("Two tiers: auto-remove at high precision, review the band below")
    ax1.legend(fontsize=8)

    ax2.plot(recall * 100, prevalence_after * 1e4, color="C3", lw=2)
    ax2.set_xlabel("recall of the enforcement system (%)")
    ax2.set_ylabel("prevalence: violating views per 10k (illustrative)")
    ax2.set_title("Prevalence is the user-facing metric, not the model's recall")
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
