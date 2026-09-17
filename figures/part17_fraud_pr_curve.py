"""Fraud detection: precision-recall at 0.2% prevalence, with operating points and a cost-weighted optimum."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_fraud_pr_curve.png"


def main() -> None:
    rng = np.random.default_rng(3)
    n = 500_000
    prevalence = 0.002
    y = rng.random(n) < prevalence
    score = np.where(y, rng.normal(2.2, 1.0, n), rng.normal(0.0, 1.0, n))
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    fpr = fp / (~y).sum()

    # Cost model: each missed fraud costs 100 units (chargeback), each false decline costs 5 units (lost sale).
    cost = 100 * (y.sum() - tp) + 5 * fp
    best = np.argmin(cost)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(recall, precision, color="C0", lw=2)
    ax1.scatter([recall[best]], [precision[best]], color="C3", zorder=3, s=50,
                label=f"cost-optimal: recall {recall[best]:.2f}, precision {precision[best]:.2f}")
    for target in (0.001, 0.005):
        k = np.searchsorted(fpr, target)
        ax1.scatter([recall[k]], [precision[k]], color="C1", zorder=3, s=40)
        ax1.annotate(f"FPR={target:.1%}: recall {recall[k]:.2f}", (recall[k], precision[k]),
                     textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax1.set_xlabel("recall (fraud caught)")
    ax1.set_ylabel("precision (of blocked, truly fraud)")
    ax1.set_title(f"PR curve at {prevalence:.1%} prevalence (AUROC would look great; this does not)")
    ax1.legend(fontsize=8, loc="lower left")

    ax2.plot(recall, cost / 1e3, color="C2", lw=2)
    ax2.axvline(recall[best], color="C3", ls="--", lw=1)
    ax2.set_xlabel("recall (moves with the threshold)")
    ax2.set_ylabel("total cost (thousands of units)")
    ax2.set_title("Threshold from the cost matrix, not from F1")
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
