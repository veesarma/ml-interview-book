"""Swin shifted windows: region ids before and after the cyclic shift, and the resulting attention mask
for the bottom-right window. Uses the book's own ``shifted_window_mask``.

Writes docs/assets/figures/part08_shifted_window_mask.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mlbook.multimodal.swin_window import region_ids, shifted_window_mask  # noqa: E402

OUT = ROOT / "docs" / "assets" / "figures" / "part08_shifted_window_mask.png"


def main() -> None:
    h = w = 8
    M, s = 4, 2
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = ListedColormap([colors[0], colors[1], colors[2], colors[3]])
    ids = region_ids(h, w, s)
    rolled = torch.roll(ids, (-s, -s), (0, 1))
    mask = shifted_window_mask(h, w, M, s)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), facecolor="white")
    for ax, grid, title in [(axes[0], ids, "before roll: region id\n(red dashes: cells that will wrap)"),
                            (axes[1], rolled, "after torch.roll(-M/2):\nwindows (black) now mix regions")]:
        ax.imshow(grid.numpy(), cmap=cmap, vmin=0, vmax=3, alpha=0.75)
        for i in range(h):
            for j in range(w):
                ax.text(j, i, str(int(grid[i, j])), ha="center", va="center", fontsize=8)
        for k in range(0, h + 1, M):
            ax.axhline(k - 0.5, color="black", linewidth=2)
            ax.axvline(k - 0.5, color="black", linewidth=2)
        if title.startswith("before"):
            ax.axhline(s - 0.5, color="red", linewidth=1.5, linestyle="--")
            ax.axvline(s - 0.5, color="red", linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    m = mask[3].numpy()  # bottom-right window
    ax.imshow(np.where(m == 0, 1.0, 0.0), cmap="Greys_r", vmin=-0.3, vmax=1.2)
    ax.set_title("mask of bottom-right window\n(16 x 16): white = attend", fontsize=9, loc="left")
    ax.set_xlabel("key token (row-major in window)", fontsize=8)
    ax.set_ylabel("query token", fontsize=8)
    ax.set_xticks(range(0, 16, 4)); ax.set_yticks(range(0, 16, 4))
    ax.tick_params(labelsize=7)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
