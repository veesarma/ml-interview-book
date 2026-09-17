"""Lift-Splat-Shoot: the per-pixel depth distribution, the frustum, and the BEV pooling.

Writes docs/assets/figures/part11_lss_frustum.png
Run from the repository root:  python figures/part11_lss_frustum.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.perception.camera_rig import make_surround_rig  # noqa: E402
from mlbook.perception.lift_splat import create_frustum, frustum_to_ego  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_lss_frustum.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), facecolor="white")

    # ---- 1. the depth distribution over bins ----------------------------------------
    ax = axes[0]
    depth_bins = np.linspace(2.0, 42.0, 21)
    alpha_sharp = np.exp(-0.5 * ((depth_bins - 14.0) / 2.0) ** 2)
    alpha_sharp /= alpha_sharp.sum()
    alpha_flat = np.exp(-0.5 * ((depth_bins - 22.0) / 9.0) ** 2)
    alpha_flat /= alpha_flat.sum()
    ax.bar(depth_bins - 0.45, alpha_sharp, width=0.9, color=colors[0], alpha=0.85, label="textured pixel (car boundary)")
    ax.bar(depth_bins + 0.45, alpha_flat, width=0.9, color=colors[1], alpha=0.85, label="untextured pixel (blank wall)")
    ax.set_xlabel("depth bin centre (m)", fontsize=9)
    ax.set_ylabel(r"$\alpha_d$", fontsize=10)
    ax.set_title("1. lift: a categorical depth per pixel", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(axis="y", color="#eeeeee", linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 2. the frustum in the ego frame --------------------------------------------
    ax = axes[1]
    cams = make_surround_rig(n_cams=3, fov_x_deg=70.0, image_hw=(128, 224))
    K = torch.tensor(np.stack([c.K for c in cams]), dtype=torch.float32)
    T = torch.tensor(np.stack([c.T_ego_from_cam for c in cams]), dtype=torch.float32)
    fr = create_frustum((128, 224), (4, 9), torch.tensor(depth_bins, dtype=torch.float32))
    pts = frustum_to_ego(fr, K, T).numpy()  # (N, D, Hf, Wf, 3)
    for i in range(3):
        p = pts[i].reshape(-1, 3)
        ax.scatter(p[:, 1], p[:, 0], s=1.6, color=colors[i], alpha=0.35, label=f"camera {i}")
    ax.scatter([0], [0], marker="s", s=60, color="black", zorder=5)
    ax.text(0, -3.5, "ego", ha="center", fontsize=8)
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.set_xlabel("y, left (m)", fontsize=9)
    ax.set_ylabel("x, forward (m)", fontsize=9)
    ax.set_title("2. frustum points, unprojected to ego", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.grid(color="#eeeeee", linewidth=0.5)

    # ---- 3. the BEV grid, coloured by how many frustum points land in each cell -------
    ax = axes[2]
    extent = (-45.0, 45.0, -45.0, 45.0)
    nx = ny = 45
    flat = pts.reshape(-1, 3)
    ix = np.floor((flat[:, 0] - extent[0]) / (extent[1] - extent[0]) * nx).astype(int)
    iy = np.floor((flat[:, 1] - extent[2]) / (extent[3] - extent[2]) * ny).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    grid = np.zeros((nx, ny))
    np.add.at(grid, (ix[ok], iy[ok]), 1.0)
    im = ax.imshow(grid, origin="lower", cmap="magma",
                   extent=(extent[2], extent[3], extent[0], extent[1]))
    ax.scatter([0], [0], marker="s", s=60, color="white", edgecolor="black", zorder=5)
    ax.set_xlabel("y, left (m)", fontsize=9)
    ax.set_ylabel("x, forward (m)", fontsize=9)
    ax.set_title("3. splat: frustum points per BEV cell", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="points pooled")

    fig.suptitle("Lift-Splat-Shoot, one camera rig, 21 depth bins", fontsize=11.5, x=0.09, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
