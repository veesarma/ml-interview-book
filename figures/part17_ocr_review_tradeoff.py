"""OCR / document extraction: confidence threshold vs straight-through rate and residual field error."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_ocr_review_tradeoff.png"


def main() -> None:
    rng = np.random.default_rng(11)
    n = 200_000
    correct = rng.random(n) < 0.94  # field-level accuracy of the extractor
    conf = np.where(correct, rng.beta(8, 1.5, n), rng.beta(2.5, 3, n))  # confidence is informative but imperfect
    ts = np.linspace(0.3, 0.99, 100)
    stp = np.array([(conf >= t).mean() for t in ts])                       # straight-through rate
    resid = np.array([((conf >= t) & ~correct).sum() / max((conf >= t).sum(), 1) for t in ts])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ts, stp * 100, color="C0", lw=2, label="straight-through (no human) %")
    ax1.plot(ts, (1 - stp) * 100, color="C1", lw=2, label="sent to human review %")
    ax1.set_xlabel("confidence threshold for auto-accept")
    ax1.set_ylabel("percent of fields")
    ax1.set_title("Threshold sets the review bill")
    ax1.legend(fontsize=8)
    ax2.plot(stp * 100, resid * 100, color="C3", lw=2)
    ax2.set_xlabel("straight-through rate (%)")
    ax2.set_ylabel("error rate among auto-accepted fields (%)")
    ax2.set_title("Residual error vs automation: pick the point from the SLA")
    ax2.axhline(0.5, color="k", ls="--", lw=1)
    ax2.text(20, 0.65, "example SLA: <0.5% error on auto-accepted fields", fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
