"""BEV queries: which cameras each query's pillar projects into, and its attention mask.

Writes docs/assets/figures/part11_bev_query.png
Run from the repository root:  python figures/part11_bev_query.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.perception.bev_query import bev_reference_points, project_points_to_cameras  # noqa: E402
from mlbook.perception.camera_rig import make_surround_rig  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_bev_query.png"
EXTENT = (-40.0, 40.0, -40.0, 40.0)
GRID = (40, 40)
IMAGE_HW = (128, 224)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cams = make_surround_rig(n_cams=6, fov_x_deg=70.0, image_hw=IMAGE_HW)
    K = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32)
    T = torch.tensor(np.stack([c.T_cam_from_ego for c in cams]), dtype=torch.float32)
    ref = bev_reference_points(EXTENT, GRID, torch.tensor([-1.0, 0.5, 2.0]))  # (X*Y, N_z, 3)
    pixels, valid = project_points_to_cameras(ref.reshape(-1, 3), K, T, IMAGE_HW)
    valid = valid.view(6, GRID[0] * GRID[1], 3).any(dim=2).numpy()  # (N, X*Y)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), facecolor="white")

    # ---- 1. how many cameras see each BEV cell ---------------------------------------
    ax = axes[0]
    n_seen = valid.sum(axis=0).reshape(GRID)
    im = ax.imshow(n_seen, origin="lower", cmap="viridis", vmin=0, vmax=2,
                   extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
    ax.scatter([0], [0], marker="s", s=55, color="white", edgecolor="black", zorder=5)
    ax.set_title("cameras whose image a pillar projects into", fontsize=9.8, loc="left")
    ax.set_xlabel("y, left (m)", fontsize=9)
    ax.set_ylabel("x, forward (m)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 1, 2])

    # ---- 2. which camera owns each cell ----------------------------------------------
    ax = axes[1]
    owner = np.where(valid.any(axis=0), valid.argmax(axis=0), -1).reshape(GRID).astype(float)
    owner[n_seen == 0] = np.nan
    cmap = matplotlib.colors.ListedColormap(colors[:6])
    im = ax.imshow(owner, origin="lower", cmap=cmap, vmin=-0.5, vmax=5.5,
                   extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
    ax.scatter([0], [0], marker="s", s=55, color="white", edgecolor="black", zorder=5)
    for i, cam in enumerate(cams):
        yaw = 2.0 * np.pi * i / 6
        ax.annotate("", xy=(14 * np.sin(yaw), 14 * np.cos(yaw)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="white", linewidth=1.4))
        ax.text(21 * np.sin(yaw), 21 * np.cos(yaw), f"cam{i}", color="white", fontsize=7.5,
                ha="center", va="center")
    ax.set_title("first camera that sees the pillar", fontsize=9.8, loc="left")
    ax.set_xlabel("y, left (m)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=range(6))

    # ---- 3. the attention mask of two example queries --------------------------------
    ax = axes[2]
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0, 9.4, "attention mask per query (6 cameras x 3 z-levels)", fontsize=9.8)
    picks = [(30, 20, "cell at x=+21 m, ahead"), (8, 20, "cell at x=-25 m, behind")]
    full = torch.stack([v for v in project_points_to_cameras(ref.reshape(-1, 3), K, T, IMAGE_HW)[1].view(6, -1, 3)])
    for row, (cx, cy, label) in enumerate(picks):
        q = cx * GRID[1] + cy
        mask = full[:, q, :].numpy()  # (6, 3)
        y0 = 6.0 - row * 4.0
        for i in range(6):
            for j in range(3):
                ax.add_patch(plt.Rectangle((0.6 + i * 1.3, y0 + j * 0.65), 1.2, 0.6,
                                           facecolor=colors[2] if mask[i, j] else "#e8e8e8",
                                           edgecolor="white"))
        ax.text(0.6, y0 + 2.2, label, fontsize=8.5)
        ax.text(0.2, y0 + 0.9, "z", fontsize=8, rotation=90, va="center")
        for i in range(6):
            ax.text(1.2 + i * 1.3, y0 - 0.45, f"c{i}", fontsize=7.5, ha="center")
        ax.text(8.6, y0 + 0.9, f"{int(mask.sum())} of 18\nkeys attended", fontsize=7.8, va="center")
    ax.text(0, 0.5, "Queries attend only where their own pillar lands. Cells no camera\n"
                    "sees keep their previous value, which is what makes the update sparse.",
            fontsize=8.2, color="#333333")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
