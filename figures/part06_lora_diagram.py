"""LoRA diagram (frozen W plus trainable B·A path) and trainable-parameter count vs rank
for a Llama-2-7B-shaped attention block, compared with full fine-tuning.

Writes docs/assets/figures/part06_lora_diagram.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_lora_diagram.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    ax1.add_patch(FancyBboxPatch((3.2, 3.2), 3.6, 2.2, boxstyle="round,pad=0.02", facecolor="#dddddd", edgecolor="none"))
    ax1.text(5.0, 4.3, "W  (out × in)\nfrozen, bf16", ha="center", va="center", fontsize=9)
    ax1.add_patch(FancyBboxPatch((3.2, 0.6), 1.4, 1.6, boxstyle="round,pad=0.02", facecolor=colors[1], edgecolor="none", alpha=0.9))
    ax1.text(3.9, 1.4, "A\n(r × in)\nrandom init", ha="center", va="center", fontsize=8)
    ax1.add_patch(FancyBboxPatch((5.4, 0.6), 1.4, 1.6, boxstyle="round,pad=0.02", facecolor=colors[1], edgecolor="none", alpha=0.9))
    ax1.text(6.1, 1.4, "B\n(out × r)\nzero init", ha="center", va="center", fontsize=8)
    for (x0, y0), (x1, y1) in [((1.0, 3.0), (3.2, 4.3)), ((1.0, 3.0), (3.2, 1.4)), ((4.6, 1.4), (5.4, 1.4)),
                               ((6.8, 4.3), (8.6, 3.0)), ((6.8, 1.4), (8.6, 3.0))]:
        ax1.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=11, linewidth=1.4))
    ax1.text(0.7, 3.0, "x", fontsize=11, ha="right", va="center")
    ax1.text(8.9, 3.0, "h = xWᵀ + (α/r)·xAᵀBᵀ", fontsize=9, ha="left", va="center")
    ax1.text(5.0, 2.55, "trainable: r·(in + out) params, gradient only flows here", ha="center", fontsize=7.5, color="#444444")
    ax1.set_title("LoRA: W' = W + (α/r) B A, rank r ≪ min(in, out)", fontsize=10, loc="left")

    d = 4096
    ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n_layers = 32
    per_layer_qv = 2 * ranks * (d + d)  # q_proj and v_proj, each r(in+out)
    per_layer_all4 = 4 * ranks * (d + d)
    full_attn = n_layers * 4 * d * d
    ax2.plot(ranks, n_layers * per_layer_qv / 1e6, marker="o", color=colors[0], linewidth=1.8, label="LoRA on q,v (32 layers)")
    ax2.plot(ranks, n_layers * per_layer_all4 / 1e6, marker="o", color=colors[2], linewidth=1.8, label="LoRA on q,k,v,o (32 layers)")
    ax2.axhline(full_attn / 1e6, color=colors[3], linewidth=1.5, linestyle="--")
    ax2.text(ranks[0], full_attn / 1e6 * 1.25, "full fine-tune of the same 4 matrices: 2,147 M params", fontsize=7.5, color=colors[3])
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("rank r")
    ax2.set_ylabel("trainable parameters (millions)")
    ax2.set_title("Trainable parameters vs rank (d_model = 4096)", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.grid(color="#eeeeee", linewidth=0.5, which="both")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
