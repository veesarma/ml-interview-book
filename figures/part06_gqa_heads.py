"""MHA vs GQA vs MQA head sharing diagram with KV-cache bytes per token per layer.

Writes docs/assets/figures/part06_gqa_heads.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_gqa_heads.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    H, d_head = 8, 128
    configs = [("MHA  (H_kv = 8)", 8), ("GQA  (H_kv = 2)", 2), ("MQA  (H_kv = 1)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor="white")
    for ax, (name, h_kv) in zip(axes, configs):
        ax.set_xlim(0, H)
        ax.set_ylim(0, 3.2)
        ax.axis("off")
        group = H // h_kv
        for h in range(H):
            ax.add_patch(FancyBboxPatch((h + 0.1, 2.2), 0.8, 0.7, boxstyle="round,pad=0.01", facecolor=colors[0], edgecolor="none"))
            ax.text(h + 0.5, 2.55, f"Q{h}", ha="center", va="center", fontsize=7, color="white")
            kv = h // group
            ax.plot([h + 0.5, kv * group + group / 2], [2.2, 1.5], color="#777777", linewidth=0.8)
        for kv in range(h_kv):
            x0 = kv * group + 0.1
            ax.add_patch(FancyBboxPatch((x0, 0.8), group - 0.2, 0.7, boxstyle="round,pad=0.01", facecolor=colors[2], edgecolor="none"))
            label = f"K{kv}\nV{kv}" if group == 1 else f"K{kv} V{kv}"
            ax.text(x0 + (group - 0.2) / 2, 1.15, label, ha="center", va="center", fontsize=6.5 if group == 1 else 7, color="white")
        kv_bytes = 2 * h_kv * d_head * 2
        ax.text(H / 2, 0.25, f"KV per token per layer: 2·{h_kv}·{d_head}·2 B = {kv_bytes / 1024:g} KiB   (÷{H // h_kv})",
                ha="center", fontsize=7.5, color="#333333")
        ax.set_title(name, fontsize=10, loc="left")
    fig.suptitle("Query heads (top) share key/value heads (bottom): H = 8, d_head = 128, bf16", fontsize=9, y=1.02)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
