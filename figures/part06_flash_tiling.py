"""FlashAttention tiling: the (T x T) score matrix is never materialised; a query
block streams K/V blocks through SRAM keeping (m, l, acc).  Right panel: HBM traffic
of standard vs. IO-aware attention as T grows.

Writes docs/assets/figures/part06_flash_tiling.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from mlbook.llm.flash_attention import attention_hbm_bytes_flash, attention_hbm_bytes_standard  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_flash_tiling.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="white", gridspec_kw={"width_ratios": [1, 1.2]})

    n_blocks, bsz = 6, 1.0
    ax1.set_xlim(-1.6, n_blocks + 0.4)
    ax1.set_ylim(-0.8, n_blocks + 1.6)
    ax1.set_aspect("equal")
    ax1.axis("off")
    qi = 3  # the query block being processed
    for i in range(n_blocks):
        for j in range(n_blocks):
            causal_dead = j > i
            face = "#f2f2f2" if causal_dead else "white"
            if i == qi and not causal_dead:
                face = colors[0] if j == 2 else "#cfe0f5"
            ax1.add_patch(Rectangle((j, n_blocks - 1 - i), bsz, bsz, facecolor=face, edgecolor="#bbbbbb", linewidth=0.6))
    ax1.text(2.5, n_blocks - 1 - qi + 0.5, "S_blk", ha="center", va="center", fontsize=7, color="white")
    for j in range(n_blocks):
        ax1.add_patch(Rectangle((j, n_blocks + 0.3), bsz, 0.5, facecolor=colors[2] if j == 2 else "#d9efd9", edgecolor="#bbbbbb", linewidth=0.6))
    ax1.text(n_blocks / 2, n_blocks + 1.1, "K, V blocks (streamed from HBM, one at a time)", ha="center", fontsize=7.5)
    for i in range(n_blocks):
        ax1.add_patch(Rectangle((-0.9, n_blocks - 1 - i), 0.5, bsz, facecolor=colors[0] if i == qi else "#cfe0f5", edgecolor="#bbbbbb", linewidth=0.6))
    ax1.text(-1.2, n_blocks / 2, "Q blocks", rotation=90, ha="center", va="center", fontsize=7.5)
    ax1.text(n_blocks / 2, -0.45, "grey = masked by causality (skipped);  the full S matrix is never stored", ha="center", fontsize=7.2, color="#444444")
    ax1.set_title("Tiling: one (q-block, kv-block) tile in SRAM at a time", fontsize=10, loc="left")

    T = np.array([512, 1024, 2048, 4096, 8192, 16384, 32768])
    d_head = 128
    std = np.array([attention_hbm_bytes_standard(t, d_head) for t in T]) / 1e6
    fl = np.array([attention_hbm_bytes_flash(t, d_head, sram_bytes=200 * 1024) for t in T]) / 1e6
    ax2.plot(T, std, marker="o", color=colors[3], linewidth=2, label="standard: writes/reads S, P  (Θ(T²))")
    ax2.plot(T, fl, marker="o", color=colors[0], linewidth=2, label="IO-aware tiling  (Θ(T² d² / M), M = SRAM)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("sequence length T")
    ax2.set_ylabel("HBM bytes per head (MB, bf16, d_head = 128)")
    ax2.set_title("HBM traffic per attention head", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(color="#eeeeee", linewidth=0.5, which="both")
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
