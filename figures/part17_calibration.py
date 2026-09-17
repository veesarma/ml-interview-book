"""Ads calibration: a reliability diagram before and after correcting for negative down-sampling."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_calibration.png"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    rng = np.random.default_rng(1)
    n = 200_000
    logit_true = rng.normal(-3.5, 1.2, n)          # base CTR around 3%
    p_true = sigmoid(logit_true)
    y = rng.random(n) < p_true
    w = 0.1                                        # keep 10% of negatives
    # A model trained on down-sampled negatives learns logit + log(1/w); simulate that shift.
    p_model = sigmoid(logit_true + np.log(1.0 / w))
    # Correction: q = p / (p + (1 - p) / w)
    p_corr = p_model / (p_model + (1.0 - p_model) / w)

    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for p, c, label in [(p_model, "C3", "raw model on 10% negatives (over-predicts)"), (p_corr, "C2", "after q = p / (p + (1-p)/w)")]:
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(20):
            m = idx == b
            if m.sum() > 200:
                xs.append(p[m].mean())
                ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=c, lw=2, ms=5, label=label)
    ax.set_xlabel("predicted click probability")
    ax.set_ylabel("observed click rate")
    ax.set_title("Down-sampling negatives shifts the logit by log(1/w); undo it before the auction")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
