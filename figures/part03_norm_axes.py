"""Which elements share statistics under BatchNorm / LayerNorm / GroupNorm / InstanceNorm,
drawn on a (N, C, spatial) block. -> docs/assets/figures/part03_norm_axes.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N, C, S = 4, 6, 5  # batch, channels, flattened spatial (H*W)
panels = [
    ("BatchNorm\nstats over (N, H, W) per channel", lambda n, c, s: c == 1),
    ("LayerNorm\nstats over (C, H, W) per sample", lambda n, c, s: n == 1),
    ("GroupNorm (G=2)\nstats over (C/G, H, W) per sample", lambda n, c, s: (n == 1) and (c < C // 2)),
    ("InstanceNorm\nstats over (H, W) per sample & channel", lambda n, c, s: (n == 1) and (c == 1)),
]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), facecolor="white")
for ax, (title, member) in zip(axes, panels):
    ax.set_title(title, fontsize=9)
    # draw an oblique "cube": rows = channels (C), cols = batch (N), depth = spatial (S)
    for s in reversed(range(S)):
        off = 0.35 * s
        for n in range(N):
            for c in range(C):
                x0, y0 = n + off, c + off
                color = "#2563eb" if member(n, c, s) else "#e5e7eb"
                ax.add_patch(plt.Rectangle((x0, y0), 1, 1, fc=color, ec="white", lw=0.6, alpha=0.95 if s == 0 else 0.7))
    ax.set_xlim(-0.2, N + 0.35 * S + 0.5)
    ax.set_ylim(-0.2, C + 0.35 * S + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("N (batch)  ->", fontsize=8)
    ax.set_ylabel("C (channels)  ->", fontsize=8)
    ax.text(N + 0.35 * S - 0.5, C + 0.35 * S + 0.05, "H*W", fontsize=7, color="#374151")
fig.suptitle("Blue = one normalisation group (mean and variance computed over the blue cells)", fontsize=9)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_norm_axes.png", dpi=150, bbox_inches="tight", facecolor="white")
