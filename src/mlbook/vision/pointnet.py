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
        super().__init__()
        layers: list[nn.Module] = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(c_in, c_out, kernel_size=1), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, C_out, N)


class TNet(nn.Module):
    """Predicts a ``k×k`` transform from the cloud: ``T = I + MLP(maxpool(sharedMLP(x)))``.

    Initialised (via the identity add) to the identity so training starts as a no-op.
    (B, k, N) → (B, k, k).
    """

    def __init__(self, k: int = 3, width: int = 64) -> None:
        super().__init__()
        self.k = k
        self.features = SharedMLP([k, width, 2 * width, 4 * width])
        self.regress = nn.Sequential(
            nn.Linear(4 * width, 2 * width), nn.BatchNorm1d(2 * width), nn.ReLU(inplace=True),
            nn.Linear(2 * width, width), nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.Linear(width, k * k),
        )
        nn.init.zeros_(self.regress[-1].weight)
        nn.init.zeros_(self.regress[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)  # (B, 4w, N)
        g = f.max(dim=2).values  # (B, 4w)  symmetric aggregation
        delta = self.regress(g).reshape(-1, self.k, self.k)  # (B, k, k)
        eye = torch.eye(self.k, device=x.device, dtype=x.dtype)[None]  # (1, k, k)
        return eye + delta  # (B, k, k)


class TinyPointNet(nn.Module):
    """Classification PointNet: input T-Net → shared MLP → feature T-Net → shared MLP → max → MLP.

    (B, N, 3) → logits (B, K).  ``forward`` also returns the 64×64 feature transform so
    the caller can add the orthogonality regulariser ``‖I − A Aᵀ‖²_F``.
    """

    def __init__(self, num_classes: int = 4, width: int = 32) -> None:
        super().__init__()
        self.input_tnet = TNet(k=3, width=width)
        self.mlp1 = SharedMLP([3, width, width])
        self.feature_tnet = TNet(k=width, width=width)
        self.mlp2 = SharedMLP([width, 2 * width, 8 * width])
        self.head = nn.Sequential(nn.Linear(8 * width, 4 * width), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(4 * width, num_classes))

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = points.transpose(1, 2)  # (B, 3, N)  channels-first for Conv1d
        T_in = self.input_tnet(x)  # (B, 3, 3)
        x = torch.bmm(T_in, x)  # (B, 3, N)  align input cloud
        x = self.mlp1(x)  # (B, w, N)
        T_feat = self.feature_tnet(x)  # (B, w, w)
        x = torch.bmm(T_feat, x)  # (B, w, N)  align features
        x = self.mlp2(x)  # (B, 8w, N)
        global_feature = x.max(dim=2).values  # (B, 8w)  order-independent summary
        return self.head(global_feature), T_feat  # (B, K), (B, w, w)


def feature_transform_regularizer(T: torch.Tensor) -> torch.Tensor:
    """``mean_b ‖I − T_b T_bᵀ‖_F²`` — keeps the feature transform near-orthogonal. ``T``: (B, k, k)."""
    k = T.shape[1]
    eye = torch.eye(k, device=T.device, dtype=T.dtype)[None]  # (1, k, k)
    diff = eye - torch.bmm(T, T.transpose(1, 2))  # (B, k, k)
    return (diff**2).sum(dim=(1, 2)).mean()  # scalar


def synthetic_point_clouds(n: int, num_points: int = 128, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Four shapes sampled as point clouds: sphere surface, cube surface, flat disc, line segment.

    Returns:
        points (n, N, 3) randomly rotated about z and jittered; labels (n,).
    """
    g = generator if generator is not None else torch.Generator().manual_seed(0)
    labels = torch.randint(0, 4, (n,), generator=g)  # (n,)
    clouds = torch.empty(n, num_points, 3)  # (n, N, 3)
    for i in range(n):
        y = labels[i].item()
        if y == 0:  # sphere surface
            p = torch.randn(num_points, 3, generator=g)  # (N, 3)
            p = p / p.norm(dim=1, keepdim=True)  # (N, 3)
        elif y == 1:  # cube surface: random face, uniform on it
            p = torch.rand(num_points, 3, generator=g) * 2 - 1  # (N, 3)
            axis = torch.randint(0, 3, (num_points,), generator=g)  # (N,)
            sign = torch.randint(0, 2, (num_points,), generator=g).to(torch.float32) * 2 - 1  # (N,)
            p[torch.arange(num_points), axis] = sign
        elif y == 2:  # flat disc in the xy-plane
            r = torch.sqrt(torch.rand(num_points, generator=g))  # (N,)
            th = 2 * torch.pi * torch.rand(num_points, generator=g)  # (N,)
            p = torch.stack([r * torch.cos(th), r * torch.sin(th), torch.zeros(num_points)], dim=1)  # (N, 3)
        else:  # line segment along x
            t = torch.rand(num_points, generator=g) * 2 - 1  # (N,)
            p = torch.stack([t, torch.zeros(num_points), torch.zeros(num_points)], dim=1)  # (N, 3)
        angle = 2 * torch.pi * torch.rand(1, generator=g).item()
        Rz = torch.tensor([[torch.cos(torch.tensor(angle)), -torch.sin(torch.tensor(angle)), 0.0],
                           [torch.sin(torch.tensor(angle)), torch.cos(torch.tensor(angle)), 0.0],
                           [0.0, 0.0, 1.0]])  # (3, 3)
        clouds[i] = p @ Rz.T + 0.02 * torch.randn(num_points, 3, generator=g)  # (N, 3)
    return clouds, labels
