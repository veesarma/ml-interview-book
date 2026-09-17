"""Reliability diagram before and after temperature scaling on synthetic
over-confident logits. Writes docs/assets/figures/part13_reliability_diagram.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.reliability.calibration import TemperatureScaling, expected_calibration_error  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_reliability_diagram.png"


def main() -> None:
    torch.manual_seed(0)
    N, K = 6000, 10
    true_logits = torch.randn(N, K) * 1.5
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    over = true_logits * 2.5  # an over-confident model (too-sharp softmax)
    val, test = slice(0, N // 2), slice(N // 2, N)
    ts = TemperatureScaling().fit(over[val], labels[val])
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, logits, name in ((axes[0], over[test], "before"), (axes[1], ts(over[test]).detach(), f"after (T={ts.temperature:.2f})")):
        probs = torch.softmax(logits, 1).numpy()
        ece, _, bins = expected_calibration_error(probs, labels[test].numpy(), n_bins=10)
        centres = (bins["edges"][:-1] + bins["edges"][1:]) / 2
        ax.bar(centres, np.nan_to_num(bins["acc"]), width=0.1, edgecolor="k", alpha=0.8, label="accuracy in bin")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.set(xlabel="confidence (max softmax)", title=f"{name}: ECE = {ece:.3f}", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    axes[0].set_ylabel("empirical accuracy")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
