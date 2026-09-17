"""Two-dimensional toy distributions shared by the GAN, DDPM and flow-matching chapters."""

from __future__ import annotations

import math

import torch


def gaussian_mixture_2d(n: int, n_modes: int = 8, radius: float = 2.0, std: float = 0.15,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """``n`` points from ``n_modes`` Gaussians placed evenly on a circle of ``radius``.

    Returns:
        x: (N, 2)  samples, one row per point.
    """
    angles = torch.arange(n_modes) * (2 * math.pi / n_modes)             # (M,)
    centers = radius * torch.stack([angles.cos(), angles.sin()], dim=1)  # (M, 2)
    which = torch.randint(0, n_modes, (n,), generator=generator)         # (N,)
    noise = std * torch.randn(n, 2, generator=generator)                 # (N, 2)
    x = centers[which] + noise                                           # (N, 2)
    return x


def gaussian_mixture_2d_labeled(n: int, n_modes: int = 8, radius: float = 2.0, std: float = 0.15,
                                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Same mixture, also returning the mode index as a class label (for conditional models).

    Returns:
        x: (N, 2)   samples.
        y: (N,)     integer mode index in ``[0, n_modes)``.
    """
    angles = torch.arange(n_modes) * (2 * math.pi / n_modes)             # (M,)
    centers = radius * torch.stack([angles.cos(), angles.sin()], dim=1)  # (M, 2)
    y = torch.randint(0, n_modes, (n,), generator=generator)             # (N,)
    x = centers[y] + std * torch.randn(n, 2, generator=generator)        # (N, 2)
    return x, y


def two_moons(n: int, noise: float = 0.08, generator: torch.Generator | None = None) -> torch.Tensor:
    """The classic two interleaving half-circles, centred at the origin.

    Returns:
        x: (N, 2).
    """
    n_upper = n // 2
    n_lower = n - n_upper
    t_up = math.pi * torch.rand(n_upper, generator=generator)            # (N/2,)
    t_lo = math.pi * torch.rand(n_lower, generator=generator)            # (N/2,)
    upper = torch.stack([t_up.cos(), t_up.sin()], dim=1)                 # (N/2, 2)
    lower = torch.stack([1.0 - t_lo.cos(), 0.5 - t_lo.sin()], dim=1)     # (N/2, 2)
    x = torch.cat([upper, lower], dim=0)                                 # (N, 2)
    x = x + noise * torch.randn(n, 2, generator=generator)               # (N, 2)
    x = x - torch.tensor([0.5, 0.25])                                    # (N, 2) recentre
    return x


def distance_to_mixture_modes(x: torch.Tensor, n_modes: int = 8, radius: float = 2.0) -> torch.Tensor:
    """Euclidean distance of every point to its nearest mixture centre (a manifold-proximity metric).

    Args:
        x: (N, 2).
    Returns:
        d: (N,)  distance to the closest of the ``n_modes`` centres.
    """
    angles = torch.arange(n_modes) * (2 * math.pi / n_modes)             # (M,)
    centers = radius * torch.stack([angles.cos(), angles.sin()], dim=1)  # (M, 2)
    diff = x[:, None, :] - centers[None, :, :]                           # (N, M, 2)
    d = diff.norm(dim=-1).min(dim=1).values                              # (N,)
    return d
