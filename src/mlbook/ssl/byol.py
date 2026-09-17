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

    def __init__(self, encoder: nn.Module, d_h: int, d_z: int = 32, d_hidden: int = 64, tau: float = 0.99) -> None:
        super().__init__()
        self.online_encoder = encoder
        self.online_projector = nn.Sequential(nn.Linear(d_h, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_z))
        self.predictor = nn.Sequential(nn.Linear(d_z, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_z))
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        for p in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            p.requires_grad_(False)
        self.tau = tau

    def online(self, x: torch.Tensor) -> torch.Tensor:
        h = self.online_encoder(x)                                # (B, d_h)
        z = self.online_projector(h)                              # (B, d_z)
        p = self.predictor(z)                                     # (B, d_z)
        return p

    @torch.no_grad()
    def target(self, x: torch.Tensor) -> torch.Tensor:
        h = self.target_encoder(x)                                # (B, d_h)
        z = self.target_projector(h)                              # (B, d_z)
        return z

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Symmetrised BYOL loss for two views (B, d_in) each.  Returns a scalar."""
        p1 = self.online(x1)                                      # (B, d_z)
        p2 = self.online(x2)                                      # (B, d_z)
        z1 = self.target(x1).detach()                             # (B, d_z)  stop-gradient
        z2 = self.target(x2).detach()                             # (B, d_z)
        return byol_regression_loss(p1, z2) + byol_regression_loss(p2, z1)

    @torch.no_grad()
    def update_target(self) -> None:
        """ξ ← τ ξ + (1 − τ) θ for encoder and projector parameters."""
        ema_update(self.target_encoder, self.online_encoder, self.tau)
        ema_update(self.target_projector, self.online_projector, self.tau)


def byol_regression_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``mean_i || p_i/||p_i|| − z_i/||z_i|| ||² = mean_i (2 − 2 cos(p_i, z_i))``.  p, z: (B, d)."""
    p_n = F.normalize(p, dim=1)                                   # (B, d)
    z_n = F.normalize(z, dim=1)                                   # (B, d)
    return ((p_n - z_n) ** 2).sum(dim=1).mean()


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """In-place ``target ← τ target + (1 − τ) online`` over matching parameters."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.mul_(tau).add_(p_o.detach(), alpha=1.0 - tau)


def byol_step(model: BYOL, opt: torch.optim.Optimizer, x1: torch.Tensor, x2: torch.Tensor) -> float:
    """One BYOL update: loss → backward (online branch only) → optimiser step → EMA target update."""
    loss = model(x1, x2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    model.update_target()
    return float(loss.detach())
