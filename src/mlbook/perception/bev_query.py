"""BEVFormer-style BEV queries with spatial cross-attention, and PETR 3D position encoding.

**BEV queries** (BEVFormer, Li et al., ECCV 2022).  A learnable grid of queries
``Q ∈ R^{X·Y × d}`` lives in the ego frame.  Each query owns a pillar of ``N_z`` reference
points ``(x, y, z_k)``; the points are projected into every camera with ``K`` and
``T_cam_from_ego``; image features are bilinearly sampled at the hit pixels; the query then
attends over its own sampled features (a *sparse* cross-attention — the query never looks
at pixels its pillar does not project to).  Real BEVFormer uses deformable attention
(learned pixel offsets around each hit); here the sampling locations are the projections
themselves, which is the part that carries the geometry.

**PETR** (Liu et al., ECCV 2022) goes the other way: instead of moving queries into the
images it gives every image token a 3D position embedding computed from the frustum
points it could correspond to, so a plain DETR decoder can attend globally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def bev_reference_points(bev_extent: tuple[float, float, float, float], bev_hw: tuple[int, int],
                         z_levels: torch.Tensor) -> torch.Tensor:
    """Cell-centre pillars: (X·Y, N_z, 3) ego xyz for a BEV grid laid out as (X, Y)."""
    x_min, x_max, y_min, y_max = bev_extent
    nx, ny = bev_hw
    xs = x_min + (torch.arange(nx, dtype=torch.float32) + 0.5) * (x_max - x_min) / nx  # (X,)
    ys = y_min + (torch.arange(ny, dtype=torch.float32) + 0.5) * (y_max - y_min) / ny  # (Y,)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")  # (X, Y) each
    xy = torch.stack([gx, gy], dim=-1).view(-1, 1, 2).expand(-1, z_levels.numel(), -1)  # (X·Y, N_z, 2)
    z = z_levels.view(1, -1, 1).expand(xy.shape[0], -1, -1)  # (X·Y, N_z, 1)
    return torch.cat([xy, z], dim=-1)  # (X·Y, N_z, 3)


def project_points_to_cameras(points_ego: torch.Tensor, K: torch.Tensor, T_cam_from_ego: torch.Tensor,
                              image_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ego points into ``N`` cameras.

    Args:
        points_ego: (P, 3).  K: (N, 3, 3).  T_cam_from_ego: (N, 4, 4).
    Returns:
        pixels (N, P, 2) as (u, v); valid (N, P) bool (in front of the camera, inside the image).
    """
    R = T_cam_from_ego[:, :3, :3]  # (N, 3, 3)
    t = T_cam_from_ego[:, :3, 3]  # (N, 3)
    p_cam = torch.matmul(points_ego.unsqueeze(0), R.transpose(1, 2)) + t.unsqueeze(1)  # (N, P, 3)
    depth = p_cam[..., 2]  # (N, P)
    z_safe = depth.clamp(min=1e-6)  # (N, P)
    u = K[:, 0, 0].view(-1, 1) * p_cam[..., 0] / z_safe + K[:, 0, 2].view(-1, 1)  # (N, P)
    v = K[:, 1, 1].view(-1, 1) * p_cam[..., 1] / z_safe + K[:, 1, 2].view(-1, 1)  # (N, P)
    h, w = image_hw
    valid = (depth > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)  # (N, P)
    return torch.stack([u, v], dim=-1), valid  # (N, P, 2), (N, P)


class BEVQueryCrossAttention(nn.Module):
    """Each BEV query attends to the image features its reference pillar projects onto.

    forward: queries (B, Nq, d), image feats (B, N, C, Hf, Wf) → (B, Nq, d).
    """

    def __init__(self, d_model: int, c_img: int, image_hw: tuple[int, int]):
        super().__init__()
        self.image_hw = image_hw
        self.q_proj = nn.Linear(d_model, d_model)  # queries → Q
        self.k_proj = nn.Linear(c_img, d_model)  # sampled image features → K
        self.v_proj = nn.Linear(c_img, d_model)  # sampled image features → V
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def sample_image_features(self, feats: torch.Tensor, ref_points: torch.Tensor, K: torch.Tensor,
                              T_cam_from_ego: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear samples at every (camera, query, z-level) hit.

        Args:
            feats: (B, N, C, Hf, Wf).  ref_points: (Nq, N_z, 3).
        Returns:
            sampled (B, Nq, N·N_z, C); valid (Nq, N·N_z) bool.
        """
        b, n, c, hf, wf = feats.shape
        nq, nz, _ = ref_points.shape
        pixels, valid = project_points_to_cameras(ref_points.view(-1, 3), K, T_cam_from_ego, self.image_hw)  # (N, Nq·N_z, 2), (N, Nq·N_z)
        h, w = self.image_hw
        # Pixel → normalised [−1, 1] for grid_sample (align_corners=False: −1 is the left image edge).
        gx = pixels[..., 0] / w * 2.0 - 1.0  # (N, Nq·N_z)
        gy = pixels[..., 1] / h * 2.0 - 1.0  # (N, Nq·N_z)
        grid = torch.stack([gx, gy], dim=-1).view(1, n, 1, nq * nz, 2).expand(b, -1, -1, -1, -1)  # (B, N, 1, Nq·N_z, 2)
        grid = grid.reshape(b * n, 1, nq * nz, 2)  # (B·N, 1, Nq·N_z, 2)
        sampled = F.grid_sample(feats.reshape(b * n, c, hf, wf), grid, align_corners=False, padding_mode="zeros")  # (B·N, C, 1, Nq·N_z)
        sampled = sampled.view(b, n, c, nq, nz).permute(0, 3, 1, 4, 2).reshape(b, nq, n * nz, c)  # (B, Nq, N·N_z, C)
        valid = valid.view(n, nq, nz).permute(1, 0, 2).reshape(nq, n * nz)  # (Nq, N·N_z)
        return sampled, valid

    def forward(self, queries: torch.Tensor, feats: torch.Tensor, ref_points: torch.Tensor, K: torch.Tensor,
                T_cam_from_ego: torch.Tensor) -> torch.Tensor:
        """Masked softmax attention of each query over its own hits; residual added.

        Queries whose pillar hits no camera are returned unchanged (BEVFormer does the same).
        """
        sampled, valid = self.sample_image_features(feats, ref_points, K, T_cam_from_ego)  # (B, Nq, S, C), (Nq, S)
        q = self.q_proj(queries)  # (B, Nq, d)
        k = self.k_proj(sampled)  # (B, Nq, S, d)
        v = self.v_proj(sampled)  # (B, Nq, S, d)
        scores = (k * q.unsqueeze(2)).sum(-1) * self.scale  # (B, Nq, S)  q_i · k_{is} / sqrt(d)
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))  # (B, Nq, S)
        any_hit = valid.any(dim=1)  # (Nq,)
        attn = torch.softmax(scores, dim=-1)  # (B, Nq, S)  NaN where no hits
        attn = torch.where(any_hit.view(1, -1, 1), attn, torch.zeros_like(attn))  # (B, Nq, S)
        out = (attn.unsqueeze(-1) * v).sum(2)  # (B, Nq, d)
        delta = self.out_proj(out) * any_hit.view(1, -1, 1).to(out.dtype)  # (B, Nq, d) zero update for unseen pillars
        return queries + delta  # (B, Nq, d)


class PETRPositionEncoder(nn.Module):
    """3D position embedding for image tokens: frustum xyz along ``D`` depths → MLP → d.

    forward(K (N, 3, 3), T_ego_from_cam (N, 4, 4)) → (N, Hf, Wf, d).  The MLP input for a
    pixel is its ``D`` ego-frame points (normalised into the BEV extent) concatenated, so
    two pixels in different cameras that look at the same ray region get similar embeddings.
    """

    def __init__(self, d_model: int, image_hw: tuple[int, int], feature_hw: tuple[int, int], depth_bins: torch.Tensor,
                 extent: tuple[float, float, float, float, float, float]):
        super().__init__()
        from mlbook.perception.lift_splat import create_frustum  # local import keeps modules self-contained

        self.register_buffer("frustum", create_frustum(image_hw, feature_hw, depth_bins))  # (D, Hf, Wf, 3)
        self.register_buffer("lo", torch.tensor(extent[0::2], dtype=torch.float32))  # (3,) x_min, y_min, z_min
        self.register_buffer("hi", torch.tensor(extent[1::2], dtype=torch.float32))  # (3,) x_max, y_max, z_max
        d = depth_bins.numel()
        self.mlp = nn.Sequential(nn.Linear(3 * d, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, K: torch.Tensor, T_ego_from_cam: torch.Tensor) -> torch.Tensor:
        from mlbook.perception.lift_splat import frustum_to_ego

        p_ego = frustum_to_ego(self.frustum, K, T_ego_from_cam)  # (N, D, Hf, Wf, 3)
        p_norm = (p_ego - self.lo) / (self.hi - self.lo)  # (N, D, Hf, Wf, 3) in [0, 1] inside the extent
        n, d, hf, wf, _ = p_norm.shape
        flat = p_norm.permute(0, 2, 3, 1, 4).reshape(n, hf, wf, d * 3)  # (N, Hf, Wf, 3·D)
        return self.mlp(flat)  # (N, Hf, Wf, d_model)
