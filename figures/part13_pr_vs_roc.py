"""The same scorer under balanced (1:1) and imbalanced (1:100) classes: ROC is
unchanged, the PR curve collapses. Writes docs/assets/figures/part13_pr_vs_roc.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.evaluation.classification_metrics import average_precision, pr_curve, roc_auc, roc_curve  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_pr_vs_roc.png"


def main() -> None:
    rng = np.random.default_rng(0)
    n_pos = 2000
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(9, 4))
    for ratio in (1, 100):
        n_neg = n_pos * ratio
        s_pos = rng.normal(1.5, 1.0, n_pos)
        s_neg = rng.normal(0.0, 1.0, n_neg)
        y = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        s = np.r_[s_pos, s_neg]
        fpr, tpr, _ = roc_curve(y, s)
        p, r = pr_curve(y, s)
        ax_roc.plot(fpr, tpr, label=f"1:{ratio}  AUC={roc_auc(y, s):.3f}")
        ax_pr.plot(r, p, label=f"1:{ratio}  AP={average_precision(y, s):.3f}")
    ax_roc.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax_roc.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC: insensitive to prevalence")
    ax_pr.set(xlabel="recall", ylabel="precision", title="PR: exposes the false-positive flood", ylim=(0, 1.02))
    ax_roc.legend()
    ax_pr.legend()
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
