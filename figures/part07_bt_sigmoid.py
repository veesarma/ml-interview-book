"""Bradley-Terry: P(y_w > y_l) = sigma(r_w - r_l) and the pairwise loss -log sigma(gap),
with and without a margin. Writes docs/assets/figures/part07_bt_sigmoid.png.
Run from the repository root:  python figures/part07_bt_sigmoid.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part07_bt_sigmoid.png"


def main() -> None:
    gap = np.linspace(-6, 6, 400)  # r_w - r_l
    sigma = 1.0 / (1.0 + np.exp(-gap))
    loss = -np.log(sigma)
    loss_margin = -np.log(1.0 / (1.0 + np.exp(-(gap - 1.0))))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor="white")
    ax = axes[0]
    ax.plot(gap, sigma, linewidth=2)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_xlabel("reward gap  $r(x, y_w) - r(x, y_l)$")
    ax.set_ylabel("$P(y_w \\succ y_l)$")
    ax.set_title("Bradley-Terry probability = logistic of the gap", fontsize=10, loc="left")
    ax = axes[1]
    ax.plot(gap, loss, linewidth=2, label="$-\\log\\sigma(\\Delta r)$")
    ax.plot(gap, loss_margin, linewidth=2, linestyle="--", label="$-\\log\\sigma(\\Delta r - m)$, m = 1")
    ax.plot(gap, np.maximum(0, -gap), linewidth=1, color="#666666", linestyle=":", label="asymptote $\\max(0, -\\Delta r)$")
    ax.set_xlabel("reward gap  $\\Delta r$")
    ax.set_ylabel("pairwise loss")
    ax.set_ylim(0, 6.5)
    ax.set_title("Loss saturates once the pair is well separated", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False)
    for a in axes:
        a.grid(color="#eeeeee", linewidth=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
