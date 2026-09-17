# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/pointnet.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pointnet -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/pointnet --force

"""PointNet (Qi et al. 2017) in PyTorch: shared MLP + max-pool = permutation invariance.

Input point cloud ``(B, N, 3)``; the network is invariant to the order of the ``N`` points
because every per-point operation is shared and the only cross-point op is a symmetric
max-pool.  ``TNet`` predicts a ``k×k`` alignment matrix (input/feature transform).
"""
from __future__ import annotations
import torch
from torch import nn

class SharedMLP(nn.Module):
    """Per-point MLP implemented as 1×1 ``Conv1d`` layers: (B, C_in, N) → (B, C_out, N)."""

    def __init__(self, channels: list[int]) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TNet(nn.Module):
    """Predicts a ``k×k`` transform from the cloud: ``T = I + MLP(maxpool(sharedMLP(x)))``.

    Initialised (via the identity add) to the identity so training starts as a no-op.
    (B, k, N) → (B, k, k).
    """

    def __init__(self, k: int=3, width: int=64) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyPointNet(nn.Module):
    """Classification PointNet: input T-Net → shared MLP → feature T-Net → shared MLP → max → MLP.

    (B, N, 3) → logits (B, K).  ``forward`` also returns the 64×64 feature transform so
    the caller can add the orthogonality regulariser ``‖I − A Aᵀ‖²_F``.
    """

    def __init__(self, num_classes: int=4, width: int=32) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def feature_transform_regularizer(T: torch.Tensor) -> torch.Tensor:
    """``mean_b ‖I − T_b T_bᵀ‖_F²`` — keeps the feature transform near-orthogonal. ``T``: (B, k, k)."""
    raise NotImplementedError('TODO: implement feature_transform_regularizer (see the reference in src/mlbook)')

def synthetic_point_clouds(n: int, num_points: int=128, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Four shapes sampled as point clouds: sphere surface, cube surface, flat disc, line segment.

    Returns:
        points (n, N, 3) randomly rotated about z and jittered; labels (n,).
    """
    raise NotImplementedError('TODO: implement synthetic_point_clouds (see the reference in src/mlbook)')
