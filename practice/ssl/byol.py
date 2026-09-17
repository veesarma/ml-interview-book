# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/ssl/byol.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k byol -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py ssl/byol --force

"""BYOL: bootstrap your own latent — no negatives, an EMA target network, and a predictor with stop-grad.

online:  x_1 → f_θ → g_θ → q_θ  gives  p_1 (B, d)
target:  x_2 → f_ξ → g_ξ        gives  z_2 (B, d),  with ξ ← τ ξ + (1 − τ) θ  (no gradient)
loss:    || p̄_1 − z̄_2 ||²  =  2 − 2 cos(p_1, z_2),  symmetrised over the two views.
"""
from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
from torch import nn

class BYOL(nn.Module):
    """Online encoder+projector+predictor and an EMA copy of encoder+projector.

    ``encoder``: any module (B, d_in) -> (B, d_h).  Projector and predictor are 2-layer MLPs.
    """

    def __init__(self, encoder: nn.Module, d_h: int, d_z: int=32, d_hidden: int=64, tau: float=0.99) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def online(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement online (see the reference in src/mlbook)')

    @torch.no_grad()
    def target(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement target (see the reference in src/mlbook)')

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Symmetrised BYOL loss for two views (B, d_in) each.  Returns a scalar."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    @torch.no_grad()
    def update_target(self) -> None:
        """ξ ← τ ξ + (1 − τ) θ for encoder and projector parameters."""
        raise NotImplementedError('TODO: implement update_target (see the reference in src/mlbook)')

def byol_regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``mean_i || p_i/||p_i|| − z_i/||z_i|| ||² = mean_i (2 − 2 cos(p_i, z_i))``.  p, z: (B, d)."""
    raise NotImplementedError('TODO: implement byol_regression_loss (see the reference in src/mlbook)')

@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """In-place ``target ← τ target + (1 − τ) online`` over matching parameters."""
    raise NotImplementedError('TODO: implement ema_update (see the reference in src/mlbook)')

def byol_step(model: BYOL, opt: torch.optim.Optimizer, x1: torch.Tensor, x2: torch.Tensor) -> float:
    """One BYOL update: loss → backward (online branch only) → optimiser step → EMA target update."""
    raise NotImplementedError('TODO: implement byol_step (see the reference in src/mlbook)')
