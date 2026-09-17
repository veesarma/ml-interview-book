"""3D semantic occupancy head with ego-motion-aligned temporal fusion (PyTorch).

The model estimates ``p(o_{xyz} = k | x_{1:t})`` for every voxel of a grid around the ego,
where ``k`` ranges over {free, semantic classes}.  The head here is the cheap
"channel-to-height" design (FlashOcc): a 2D conv on BEV features emits ``K·Z`` channels
that are reshaped to ``(K, Z)`` per cell — no 3D convolutions.

Temporal fusion: the previous belief, expressed in the previous ego frame, is warped into
the current frame with ``warp_bev`` (planar ego motion; every height slice moves the same
way), then blended with the current evidence by a learned gate,

    h_t = (1 − g) ⊙ warp(h_{t−1}) + g ⊙ f(x_t),   g = σ(W [warp(h_{t−1}); f(x_t)]),

i.e. a GRU-style recursive update whose fixed point is a static map and whose gate opens
where the new frame disagrees with the memory (moving objects, newly visible space).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mlbook.perception.temporal_bev import warp_bev


class OccupancyHead(nn.Module):
    """BEV features (B, C, X, Y) → voxel logits (B, K, Z, X, Y)."""

    def __init__(self, c_in: int, num_classes: int, num_z: int):
        super().__init__()
        self.k, self.z = num_classes, num_z
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c_in, num_classes * num_z, kernel_size=1),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        b, _, nx, ny = bev.shape
        logits = self.conv(bev)  # (B, K·Z, X, Y)
        return logits.view(b, self.k, self.z, nx, ny)  # (B, K, Z, X, Y) channel → height


class TemporalOccupancyFusion(nn.Module):
    """Gated fusion of the warped previous BEV memory and the current BEV features.

    forward(curr (B, C, X, Y), prev (B, C, X, Y) | None, T_curr_from_prev (B, 4, 4)) → (B, C, X, Y).
    """

    def __init__(self, c: int, bev_extent: tuple[float, float, float, float]):
        super().__init__()
        self.bev_extent = bev_extent
        self.gate = nn.Conv2d(2 * c, c, kernel_size=3, padding=1)

    def forward(self, curr: torch.Tensor, prev: torch.Tensor | None, T_curr_from_prev: torch.Tensor | None) -> torch.Tensor:
        if prev is None:
            return curr
        prev_aligned = warp_bev(prev, T_curr_from_prev, self.bev_extent)  # (B, C, X, Y)
        g = torch.sigmoid(self.gate(torch.cat([prev_aligned, curr], dim=1)))  # (B, C, X, Y) in (0, 1)
        return (1.0 - g) * prev_aligned + g * curr  # (B, C, X, Y)


def occupancy_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None,
                   ignore_index: int = 255) -> torch.Tensor:
    """Per-voxel cross-entropy.  logits (B, K, Z, X, Y), target (B, Z, X, Y) long.

    Free space is ~90 % of voxels, so ``class_weights`` (K,) (inverse-frequency) matter.
    """
    return F.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index)


def occupancy_iou(logits: torch.Tensor, target: torch.Tensor, free_class: int = 0) -> tuple[float, torch.Tensor]:
    """(binary occupied-vs-free IoU, per-class IoU (K,)) — the two numbers Occ3D reports (IoU, mIoU)."""
    k = logits.shape[1]
    pred = logits.argmax(dim=1)  # (B, Z, X, Y)
    occ_p, occ_t = pred != free_class, target != free_class  # (B, Z, X, Y) each
    binary = (occ_p & occ_t).sum().float() / (occ_p | occ_t).sum().clamp(min=1).float()
    per_class = torch.zeros(k)  # (K,)
    for c in range(k):
        p_c, t_c = pred == c, target == c
        per_class[c] = (p_c & t_c).sum().float() / (p_c | t_c).sum().clamp(min=1).float()
    return float(binary), per_class


def voxelize_points(points_ego: torch.Tensor, labels: torch.Tensor, extent: tuple[float, ...],
                    grid: tuple[int, int, int], free_class: int = 0) -> torch.Tensor:
    """Semantic points (N, 3) + labels (N,) → dense target (Z, X, Y); empty voxels = ``free_class``.

    ``extent`` = (x_min, x_max, y_min, y_max, z_min, z_max), ``grid`` = (Z, X, Y).
    Occupied voxels take the label of the last point that lands in them.
    """
    nz, nx, ny = grid
    lo = torch.tensor(extent[0::2])  # (3,)
    hi = torch.tensor(extent[1::2])  # (3,)
    size = torch.tensor([nx, ny, nz], dtype=torch.float32)  # (3,) per-axis cell counts, xyz order
    idx = torch.floor((points_ego - lo) / (hi - lo) * size).long()  # (N, 3)
    inside = ((idx >= 0) & (idx < size.long())).all(dim=1)  # (N,)
    target = torch.full((nz, nx, ny), free_class, dtype=torch.long)  # (Z, X, Y)
    ix, iy, iz = idx[inside, 0], idx[inside, 1], idx[inside, 2]  # (M,) each
    target[iz, ix, iy] = labels[inside]
    return target
