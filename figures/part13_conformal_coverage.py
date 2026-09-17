"""Empirical coverage of split conformal prediction sets over many random
calibration/test splits: centred on 1 - alpha, with the finite-sample spread.
Writes docs/assets/figures/part13_conformal_coverage.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.reliability.conformal import SplitConformalClassifier, empirical_coverage_sets  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_conformal_coverage.png"


def main() -> None:
    rng = np.random.default_rng(0)
    K, n_cal, n_test, alpha, trials = 10, 500, 2000, 0.1, 400
    y = rng.integers(0, K, n_cal + n_test)
    logits = rng.normal(size=(n_cal + n_test, K))
    logits[np.arange(len(y)), y] += 1.0
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    coverages, sizes = [], []
    for _ in range(trials):
        perm = rng.permutation(len(y))
        cal, test = perm[:n_cal], perm[n_cal:]
        c = SplitConformalClassifier(alpha).calibrate(probs[cal], y[cal])
        sets = c.predict_sets(probs[test])
        coverages.append(empirical_coverage_sets(sets, y[test]))
        sizes.append(sets.sum(1).mean())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    a1.hist(coverages, bins=25, edgecolor="k", alpha=0.8)
    a1.axvline(1 - alpha, color="C3", lw=2, label=f"target 1 - alpha = {1 - alpha}")
    a1.axvline(1 - alpha + 1 / (n_cal + 1), color="C3", ls="--", label="upper bound 1 - alpha + 1/(n+1)")
    a1.set(xlabel="empirical coverage on test split", ylabel="splits", title=f"n_cal={n_cal}, {trials} random splits")
    a1.legend(fontsize=8)
    a2.hist(sizes, bins=25, edgecolor="k", alpha=0.8, color="C1")
    a2.set(xlabel="mean prediction-set size (of 10 classes)", title="set size is the price of coverage")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
