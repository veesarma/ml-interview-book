"""Patchification: an image grid cut into P x P patches, each flattened into a token of a sequence.

Writes docs/assets/figures/part08_patchification.png. Run from the repo root:
python figures/part08_patchification.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part08_patchification.png"


def main() -> None:
    rng = np.random.default_rng(0)
    H = W = 8
    P = 4
    img = rng.random((H, W))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor="white", gridspec_kw={"width_ratios": [1, 1, 1.6]})

    ax = axes[0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for i in range(H // P):
        for j in range(W // P):
            ax.add_patch(Rectangle((j * P - 0.5, i * P - 0.5), P, P, fill=False, edgecolor=colors[i * 2 + j], linewidth=2.5))
            ax.text(j * P + P / 2 - 0.5, i * P + P / 2 - 0.5, f"patch {i * 2 + j}", ha="center", va="center", fontsize=9, color="white",
                    bbox=dict(facecolor=colors[i * 2 + j], alpha=0.85, edgecolor="none", pad=2))
    ax.set_title("image (C, H, W) = (1, 8, 8),\nP = 4", fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    for k in range(4):
        i, j = divmod(k, 2)
        vec = img[i * P:(i + 1) * P, j * P:(j + 1) * P].reshape(1, -1)
        ax.imshow(vec, cmap="gray", vmin=0, vmax=1, extent=(0, P * P, 4 - k - 0.9, 4 - k - 0.1), aspect="auto")
        ax.add_patch(Rectangle((0, 4 - k - 0.9), P * P, 0.8, fill=False, edgecolor=colors[k], linewidth=2))
        ax.text(-0.6, 4 - k - 0.5, f"{k}", ha="right", va="center", fontsize=9, color=colors[k])
    ax.set_xlim(-2, P * P); ax.set_ylim(0, 4)
    ax.set_title("flatten: N = HW/P² = 4\nvectors of P²C = 16", fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    ax = axes[2]
    d = 6
    for k in range(4):
        ax.add_patch(Rectangle((k * (d + 1), 0), d, 1, facecolor=colors[k], alpha=0.6, edgecolor="black"))
        ax.text(k * (d + 1) + d / 2, 0.5, f"z{k} ∈ ℝᵈ", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((-(d + 1), 0), d, 1, facecolor="#dddddd", edgecolor="black"))
    ax.text(-(d + 1) + d / 2, 0.5, "CLS", ha="center", va="center", fontsize=9)
    ax.text(-(d + 1), 1.35, "+ position embedding  (1+N, d)", fontsize=9)
    ax.text(-(d + 1), -0.6, "zₙ = flatten(patchₙ) E + b,   E ∈ ℝ^(P²C × d)  ==  Conv2d(kernel=P, stride=P)", fontsize=8.5)
    ax.set_xlim(-(d + 2), 4 * (d + 1)); ax.set_ylim(-1, 2)
    ax.set_title("tokens: shared linear map,\nthen the Transformer", fontsize=9, loc="left")
    ax.axis("off")

    for a, b in [(axes[0], axes[1]), (axes[1], axes[2])]:
        fig.add_artist(FancyArrowPatch((a.get_position().x1 + 0.005, 0.5), (b.get_position().x0 - 0.005, 0.5),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=15, color="black"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
