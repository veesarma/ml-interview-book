# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/toy_data.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k toy_data -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/toy_data --force

"""Two-dimensional toy distributions shared by the GAN, DDPM and flow-matching chapters."""
from __future__ import annotations
import math
import torch

def gaussian_mixture_2d(n: int, n_modes: int=8, radius: float=2.0, std: float=0.15, generator: torch.Generator | None=None) -> torch.Tensor:
    """``n`` points from ``n_modes`` Gaussians placed evenly on a circle of ``radius``.

    Returns:
        x: (N, 2)  samples, one row per point.
    """
    raise NotImplementedError('TODO: implement gaussian_mixture_2d (see the reference in src/mlbook)')

def gaussian_mixture_2d_labeled(n: int, n_modes: int=8, radius: float=2.0, std: float=0.15, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Same mixture, also returning the mode index as a class label (for conditional models).

    Returns:
        x: (N, 2)   samples.
        y: (N,)     integer mode index in ``[0, n_modes)``.
    """
    raise NotImplementedError('TODO: implement gaussian_mixture_2d_labeled (see the reference in src/mlbook)')

def two_moons(n: int, noise: float=0.08, generator: torch.Generator | None=None) -> torch.Tensor:
    """The classic two interleaving half-circles, centred at the origin.

    Returns:
        x: (N, 2).
    """
    raise NotImplementedError('TODO: implement two_moons (see the reference in src/mlbook)')

def distance_to_mixture_modes(x: torch.Tensor, n_modes: int=8, radius: float=2.0) -> torch.Tensor:
    """Euclidean distance of every point to its nearest mixture centre (a manifold-proximity metric).

    Args:
        x: (N, 2).
    Returns:
        d: (N,)  distance to the closest of the ``n_modes`` centres.
    """
    raise NotImplementedError('TODO: implement distance_to_mixture_modes (see the reference in src/mlbook)')
