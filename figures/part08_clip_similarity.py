"""CLIP's B x B similarity matrix: the diagonal holds the positives; each row and each column is a softmax.

Writes docs/assets/figures/part08_clip_similarity.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part08_clip_similarity.png"


def main() -> None:
    rng = np.random.default_rng(1)
    B = 6
    S = rng.normal(0.1, 0.12, (B, B))
    S[np.arange(B), np.arange(B)] = rng.uniform(0.55, 0.8, B)
    S[1, 3] = 0.45  # a hard negative: two similar captions
    captions = ["a dog on grass", "red sports car", "a bowl of soup", "car parked at night", "snowy mountain", "a cat asleep"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="white", gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(S, cmap="viridis", vmin=-0.2, vmax=0.9)
    for i in range(B):
        for j in range(B):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if S[i, j] < 0.55 else "black")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
    ax.set_xticks(range(B)); ax.set_xticklabels([f"t{j}" for j in range(B)], fontsize=8)
    ax.set_yticks(range(B)); ax.set_yticklabels([f"img{i}" for i in range(B)], fontsize=8)
    ax.set_xlabel("text embedding t_j", fontsize=9); ax.set_ylabel("image embedding v_i", fontsize=9)
    ax.set_title("S_ij = v_i·t_j (before /τ)\ndiagonal = the B positives, B²−B negatives", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax.annotate("hard negative", xy=(3, 1), xytext=(4.3, -0.2), fontsize=8, arrowprops=dict(arrowstyle="->"))

    ax = axes[1]
    tau = 0.07
    row = np.exp(S[1] / tau); row /= row.sum()
    col = np.exp(S[:, 1] / tau); col /= col.sum()
    x = np.arange(B)
    ax.bar(x - 0.2, row, width=0.4, label="row 1: softmax over texts (image→text)")
    ax.bar(x + 0.2, col, width=0.4, label="col 1: softmax over images (text→image)")
    ax.set_xticks(x); ax.set_xticklabels([f"{k}" for k in range(B)], fontsize=8)
    ax.set_ylabel("probability (τ = 0.07)", fontsize=9)
    ax.set_title("the two cross-entropies\nof symmetric InfoNCE", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.02, -0.04, "captions: " + "; ".join(f"t{j} = '{c}'" for j, c in enumerate(captions)), fontsize=7.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
