"""Illustrative precision/recall operating points for a payments fraud model.

This is a synthetic simulation (two score distributions, 0.5% fraud rate), NOT
Stripe data. It illustrates the framing Stripe uses publicly for Radar: a single
score with separate "block" and "review" thresholds, each a precision/recall
operating point chosen by the asymmetric cost of a false decline vs. a missed fraud.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_consumer_fraud_operating_points.png"


def main() -> None:
    rng = np.random.default_rng(0)
    n_legit, n_fraud = 200_000, 1_000                      # 0.5% base rate
    s_legit = rng.normal(0.0, 1.0, n_legit)                # (n_legit,)
    s_fraud = rng.normal(2.6, 1.1, n_fraud)                # (n_fraud,)
    scores = np.concatenate([s_legit, s_fraud])            # (N,)
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])  # (N,)
    order = np.argsort(-scores)
    tp = np.cumsum(labels[order])                          # (N,) true positives at each cut
    fp = np.cumsum(1 - labels[order])                      # (N,)
    precision = tp / (tp + fp)
    recall = tp / n_fraud
    thr = scores[order]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(recall, precision, color="C0", lw=2, label="model PR curve (synthetic)")
    for t, name, c in [(3.5, "block threshold\n(high precision, few false declines)", "C3"),
                       (2.0, "review threshold\n(higher recall, humans absorb FPs)", "C1")]:
        i = np.searchsorted(-thr, -t)
        ax.plot(recall[i], precision[i], "o", ms=9, color=c, markeredgecolor="white", markeredgewidth=1.5)
        dx, dy = (0.12, 0.08) if c == "C3" else (-0.02, 0.2)
        ax.annotate(name, (recall[i], precision[i]), xytext=(recall[i] + dx, precision[i] + dy),
                    fontsize=8.5, ha="left", arrowprops=dict(arrowstyle="-", color="0.5", lw=0.8))
    ax.set_xlabel("recall (share of fraud caught)")
    ax.set_ylabel("precision (blocked charges that were fraud)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1.0)
    ax.set_title("One score, two thresholds, asymmetric costs (illustrative)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
