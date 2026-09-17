"""Multi-stage ranking funnel: candidates per stage vs per-item cost and stage latency budget.

Numbers are illustrative orders of magnitude, not a company's reported figures.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_funnel_latency.png"

STAGES = ["Corpus", "Retrieval\n(ANN + sources)", "Pre-ranking\n(light model)", "Ranking\n(multi-task DNN)", "Re-ranking\n(policy, diversity)"]
CANDIDATES = [1e9, 5e3, 5e2, 50, 10]
FLOPS_PER_ITEM = [np.nan, 1e2, 1e5, 1e8, 1e6]  # illustrative per-item compute
LATENCY_MS = [0, 20, 15, 60, 10]


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    x = np.arange(len(STAGES))
    ax1.bar(x, CANDIDATES, color="C0", width=0.6)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(STAGES, fontsize=7)
    ax1.set_ylabel("candidates surviving the stage (log)")
    ax1.set_title("Candidates shrink ~10-100x per stage")
    for xi, c in zip(x, CANDIDATES):
        ax1.text(xi, c * 1.6, f"{c:,.0f}".replace(",", " "), ha="center", fontsize=8)
    ax1.set_ylim(1, 1e11)

    ax2b = ax2
    ax2b.bar(x[1:], LATENCY_MS[1:], color="C1", width=0.6)
    ax2b.set_xticks(x[1:])
    ax2b.set_xticklabels(STAGES[1:], fontsize=7)
    ax2b.set_ylabel("stage latency budget (ms)")
    ax2b.set_title("Latency budget: the heavy model sees only tens of items")
    for xi, ms, fl in zip(x[1:], LATENCY_MS[1:], FLOPS_PER_ITEM[1:]):
        ax2b.text(xi, ms + 2, f"{ms} ms\n~{fl:.0e} FLOP/item", ha="center", fontsize=8)
    ax2b.set_ylim(0, 90)
    for ax in (ax1, ax2b):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("The ranking funnel (illustrative magnitudes; ask the interviewer for the real ones)", fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
