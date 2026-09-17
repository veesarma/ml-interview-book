# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/bev_query.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k bev_query -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/bev_query --force

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

def bev_reference_points(bev_extent: tuple[float, float, float, float], bev_hw: tuple[int, int], z_levels: torch.Tensor) -> torch.Tensor:
    """Cell-centre pillars: (X·Y, N_z, 3) ego xyz for a BEV grid laid out as (X, Y)."""
    raise NotImplementedError('TODO: implement bev_reference_points (see the reference in src/mlbook)')

def project_points_to_cameras(points_ego: torch.Tensor, K: torch.Tensor, T_cam_from_ego: torch.Tensor, image_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ego points into ``N`` cameras.

    Args:
        points_ego: (P, 3).  K: (N, 3, 3).  T_cam_from_ego: (N, 4, 4).
    Returns:
        pixels (N, P, 2) as (u, v); valid (N, P) bool (in front of the camera, inside the image).
    """
    raise NotImplementedError('TODO: implement project_points_to_cameras (see the reference in src/mlbook)')

class BEVQueryCrossAttention(nn.Module):
    """Each BEV query attends to the image features its reference pillar projects onto.

    forward: queries (B, Nq, d), image feats (B, N, C, Hf, Wf) → (B, Nq, d).
    """

    def __init__(self, d_model: int, c_img: int, image_hw: tuple[int, int]):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def sample_image_features(self, feats: torch.Tensor, ref_points: torch.Tensor, K: torch.Tensor, T_cam_from_ego: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bilinear samples at every (camera, query, z-level) hit.

        Args:
            feats: (B, N, C, Hf, Wf).  ref_points: (Nq, N_z, 3).
        Returns:
            sampled (B, Nq, N·N_z, C); valid (Nq, N·N_z) bool.
        """
        raise NotImplementedError('TODO: implement sample_image_features (see the reference in src/mlbook)')

    def forward(self, queries: torch.Tensor, feats: torch.Tensor, ref_points: torch.Tensor, K: torch.Tensor, T_cam_from_ego: torch.Tensor) -> torch.Tensor:
        """Masked softmax attention of each query over its own hits; residual added.

        Queries whose pillar hits no camera are returned unchanged (BEVFormer does the same).
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PETRPositionEncoder(nn.Module):
    """3D position embedding for image tokens: frustum xyz along ``D`` depths → MLP → d.

    forward(K (N, 3, 3), T_ego_from_cam (N, 4, 4)) → (N, Hf, Wf, d).  The MLP input for a
    pixel is its ``D`` ego-frame points (normalised into the BEV extent) concatenated, so
    two pixels in different cameras that look at the same ray region get similar embeddings.
    """

    def __init__(self, d_model: int, image_hw: tuple[int, int], feature_hw: tuple[int, int], depth_bins: torch.Tensor, extent: tuple[float, float, float, float, float, float]):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, K: torch.Tensor, T_ego_from_cam: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
