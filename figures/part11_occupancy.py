"""Occupancy over time: ego-motion warping of the memory, and why boxes miss the long tail.

Writes docs/assets/figures/part11_occupancy.png
Run from the repository root:  python figures/part11_occupancy.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.perception.camera_rig import ego_motion_transform  # noqa: E402
from mlbook.perception.temporal_bev import warp_bev  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_occupancy.png"
EXTENT = (-16.0, 16.0, -16.0, 16.0)
N = 32


def scene(t: float) -> np.ndarray:
    """A static wall, a static irregular object, and one car moving towards the ego."""
    g = np.zeros((N, N))
    g[26:30, 6:26] = 1.0                 # a wall 10 m ahead
    g[18:21, 8:11] = 1.0                 # an irregular object: a fallen branch
    lead = int(round(22 - 2.0 * t))      # a car closing on the ego
    g[lead:lead + 3, 14:18] = 1.0
    return g


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 7.2), facecolor="white")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---- top row: three frames of raw per-frame evidence, then the fused memory -------
    memory = None
    T = torch.tensor(ego_motion_transform(dx=1.0, dy=0.0, dyaw=0.0), dtype=torch.float32).unsqueeze(0)
    for t in range(3):
        obs = scene(t)
        # per-frame evidence is partial: the far half of the wall is occluded at t=0
        vis = np.ones_like(obs)
        if t == 0:
            vis[:, 18:] = 0.0
        if t == 1:
            vis[:, :9] = 0.0
        ev = torch.tensor(obs * vis, dtype=torch.float32).view(1, 1, N, N)
        memory = ev if memory is None else 0.65 * warp_bev(memory, T, EXTENT) + 0.5 * ev
        ax = axes[0, t]
        ax.imshow(ev[0, 0].numpy(), origin="lower", cmap="Greys", vmin=0, vmax=1,
                  extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
        ax.scatter([0], [0], marker="s", s=45, color=colors[3], zorder=5)
        ax.set_ylim(-4, 16)
        ax.set_title(f"frame t = {t}: this frame only", fontsize=9.4, loc="left")
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 3]
    ax.imshow(memory[0, 0].numpy().clip(0, 1.4), origin="lower", cmap="magma",
              extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
    ax.scatter([0], [0], marker="s", s=45, color="white", edgecolor="black", zorder=5)
    ax.set_ylim(-4, 16)
    ax.set_title("fused memory, warped by ego motion", fontsize=9.4, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(-15, -3.2, "static structure sharpens,\nthe moving car smears", fontsize=8, color="white")

    # ---- bottom row: boxes vs occupancy on an irregular object -----------------------
    obs = scene(2)
    ax = axes[1, 0]
    ax.imshow(obs, origin="lower", cmap="Greys", vmin=0, vmax=1,
              extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
    ax.set_ylim(-4, 16)
    ax.set_title("ground truth occupancy", fontsize=9.4, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    ax.imshow(np.zeros_like(obs), origin="lower", cmap="Greys", vmin=0, vmax=1,
              extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
    # the box detector fires on the car, misses the wall and the branch (not in its classes)
    ax.add_patch(plt.Rectangle((-2.0, 2.0), 4.0, 3.0, fill=False, edgecolor=colors[2], lw=2.2))
    ax.text(-2.0, 5.6, "car 0.94", fontsize=8, color=colors[2])
    ax.set_ylim(-4, 16)
    ax.text(-15, -3.4, "wall: not a class, branch: not a class\nboth invisible to the planner",
            fontsize=8, color=colors[3])
    ax.set_title("closed-set box detector", fontsize=9.4, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 2]
    occ = obs.copy()
    ax.imshow(occ, origin="lower", cmap="Greys", vmin=0, vmax=1,
              extent=(EXTENT[2], EXTENT[3], EXTENT[0], EXTENT[1]))
    ax.add_patch(plt.Rectangle((-2.0, 2.0), 4.0, 3.0, fill=False, edgecolor=colors[2], lw=1.6))
    ax.set_ylim(-4, 16)
    ax.set_title("occupancy + boxes for the known classes", fontsize=9.4, loc="left")
    ax.text(-15, -3.4, "every occupied voxel is an obstacle,\nwhether or not it has a class", fontsize=8, color=colors[0])
    ax.set_xticks([]); ax.set_yticks([])

    # ---- cost of the resolution -------------------------------------------------------
    ax = axes[1, 3]
    res = np.array([0.8, 0.4, 0.2, 0.1])
    extent_m = 80.0
    height_m = 6.4
    voxels = (extent_m / res) ** 2 * (height_m / res) / 1e6
    ax.plot(res, voxels, "o-", color=colors[0], lw=1.8)
    for r, v in zip(res, voxels):
        ax.annotate(f"{v:.1f}M", (r, v), textcoords="offset points", xytext=(6, 5), fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(res)
    ax.set_xticklabels([f"{r:g}" for r in res], fontsize=8.5)
    ax.set_xlabel("voxel size (m)", fontsize=9)
    ax.set_ylabel("voxels in an 80 x 80 x 6.4 m volume (millions)", fontsize=8.5)
    ax.set_title("the cost: voxel count grows as 1/r^3", fontsize=9.4, loc="left")
    ax.grid(color="#eeeeee", lw=0.5, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
