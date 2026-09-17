"""Joint vs factorised space-time attention on a tubelet grid, and the FLOP ratio as T grows.

Writes docs/assets/figures/part08_factorised_attention.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlbook.multimodal.video_attention import attention_flops  # noqa: E402

OUT = ROOT / "docs" / "assets" / "figures" / "part08_factorised_attention.png"


def draw_grid(ax, T, h, w, highlight, title):
    c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for t in range(T):
        ox = t * (w + 1.2)
        for i in range(h):
            for j in range(w):
                key = highlight(t, i, j)
                ax.add_patch(Rectangle((ox + j, h - 1 - i), 1, 1, facecolor=c[key] if key is not None else "white",
                                       alpha=0.7 if key is not None else 1, edgecolor="grey", linewidth=0.6))
        ax.text(ox + w / 2, -0.7, f"frame t={t}", ha="center", fontsize=8)
    ax.add_patch(Rectangle((1 * (w + 1.2) + 1, h - 1 - 1), 1, 1, facecolor="black", alpha=0.9))
    ax.set_xlim(-0.3, T * (w + 1.2)); ax.set_ylim(-1.2, h + 0.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left")


def main() -> None:
    T, h, w = 3, 4, 4
    fig = plt.figure(figsize=(12, 5.4), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1])
    ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[1, 0]); ax3 = fig.add_subplot(gs[:, 1])
    draw_grid(ax1, T, h, w, lambda t, i, j: 0, "joint: the black query attends to all T·h·w tokens\n→  O((T h w)²)")
    draw_grid(ax2, T, h, w, lambda t, i, j: (1 if t == 1 else (2 if (i, j) == (1, 1) else None)),
              "factorised: spatial (orange, same frame), then temporal (green, same location)\n→  O(T (hw)²) + O(hw T²)")
    Ts = np.arange(1, 65)
    S = 16 * 16; d = 768
    joint = np.array([attention_flops(t, 16, 16, d, False) for t in Ts]) / 1e9
    fac = np.array([attention_flops(t, 16, 16, d, True) for t in Ts]) / 1e9
    ax3.plot(Ts, joint, label="joint  2N²d")
    ax3.plot(Ts, fac, label="factorised  2T(hw)²d + 2hw T²d")
    ax3.set_yscale("log"); ax3.set_xlabel("temporal tokens T′ (16×16 spatial grid, d=768)", fontsize=8.5)
    ax3.set_ylabel("attention GFLOPs per clip (QKᵀ and AV only)", fontsize=8.5)
    ax3.legend(fontsize=8, frameon=False); ax3.grid(alpha=0.3)
    ax3.set_title(f"ratio at T′=32: {joint[31] / fac[31]:.1f}×", fontsize=9, loc="left")
    for side in ("top", "right"):
        ax3.spines[side].set_visible(False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
