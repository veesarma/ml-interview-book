"""Multimodal trajectory prediction: agent encoding, social attention, WTA loss, minADE/FDE.

We model ``p(τ_{1:T} | scene)`` as a mixture of ``M`` unimodal trajectories,

    p(τ | scene) = Σ_m π_m · δ(τ − μ_m)        (a "mixture of deltas", MultiPath-style),

trained with a **winner-takes-all** loss: only the mode closest to the ground truth (by ADE)
receives regression gradient, and a cross-entropy pushes ``π`` towards that mode.  Averaging
all modes would regress to the mean of the futures — a trajectory that goes *between* the
left turn and the straight-ahead, which nobody drives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def to_agent_frame(hist: torch.Tensor, others: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agent-centric normalisation: translate + rotate so the target's last pose is the origin, heading +x.

    Args:
        hist: (B, T_h, 2) target-agent xy history; others: (B, A, T_h, 2).
    Returns:
        (hist_local (B, T_h, 2), others_local (B, A, T_h, 2), R (B, 2, 2)) where ``R`` maps
        local → world so predictions can be transformed back with ``xy_local @ R.T + origin``.
    """
    origin = hist[:, -1]  # (B, 2)
    heading = hist[:, -1] - hist[:, -2]  # (B, 2) last displacement gives the heading
    ang = torch.atan2(heading[:, 1], heading[:, 0])  # (B,)
    c, s = torch.cos(ang), torch.sin(ang)  # (B,) each
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=1)  # (B, 2, 2) local → world
    R_inv = R.transpose(1, 2)  # (B, 2, 2) world → local
    hist_local = torch.matmul(hist - origin.unsqueeze(1), R_inv.transpose(1, 2))  # (B, T_h, 2)
    others_local = torch.matmul(others - origin.view(-1, 1, 1, 2), R_inv.transpose(1, 2).unsqueeze(1))  # (B, A, T_h, 2)
    return hist_local, others_local, R


class PolylineEncoder(nn.Module):
    """VectorNet-style subgraph: per-step MLP then max-pool over time.  (B, A, T_h, 2) → (B, A, d)."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, polylines: torch.Tensor) -> torch.Tensor:
        h = self.mlp(polylines)  # (B, A, T_h, d)
        return h.max(dim=2).values  # (B, A, d) order-invariant pooling over the steps


class SocialAttention(nn.Module):
    """Target agent attends over the other agents (explicit Q/K/V).  (B, d), (B, A, d) → (B, d)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, target: torch.Tensor, others: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(target).unsqueeze(1)  # (B, 1, d)
        k = self.k_proj(others)  # (B, A, d)
        v = self.v_proj(others)  # (B, A, d)
        scores = (q * k).sum(-1) * self.scale  # (B, A)
        if valid is not None:
            scores = scores.masked_fill(~valid, float("-inf"))  # (B, A) absent agents
        attn = torch.softmax(scores, dim=-1)  # (B, A)
        attn = torch.nan_to_num(attn, nan=0.0)
        return target + (attn.unsqueeze(-1) * v).sum(1)  # (B, d) residual


class MultimodalTrajectoryHead(nn.Module):
    """Scene embedding (B, d) → ``M`` trajectories (B, M, T_f, 2) and mode logits (B, M)."""

    def __init__(self, d_model: int, num_modes: int, horizon: int):
        super().__init__()
        self.m, self.t = num_modes, horizon
        self.traj = nn.Linear(d_model, num_modes * horizon * 2)
        self.mode = nn.Linear(d_model, num_modes)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = h.shape[0]
        traj = self.traj(h).view(b, self.m, self.t, 2)  # (B, M, T_f, 2)
        return traj, self.mode(h)  # (B, M, T_f, 2), (B, M)


class TrajectoryPredictor(nn.Module):
    """History of target + others → multimodal future.  Wires the four pieces above."""

    def __init__(self, d_model: int = 64, num_modes: int = 6, horizon: int = 12):
        super().__init__()
        self.enc = PolylineEncoder(2, d_model)
        self.social = SocialAttention(d_model)
        self.head = MultimodalTrajectoryHead(d_model, num_modes, horizon)

    def forward(self, hist: torch.Tensor, others: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """hist (B, T_h, 2), others (B, A, T_h, 2) in the agent frame → ((B, M, T_f, 2), (B, M))."""
        target = self.enc(hist.unsqueeze(1)).squeeze(1)  # (B, d)
        ctx = self.enc(others)  # (B, A, d)
        return self.head(self.social(target, ctx, valid))


# ---------------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------------


def ade_per_mode(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Average displacement error of every mode: (B, M, T, 2), (B, T, 2) → (B, M)."""
    return (pred - gt.unsqueeze(1)).norm(dim=-1).mean(dim=-1)  # (B, M)


def fde_per_mode(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Final displacement error of every mode: (B, M)."""
    return (pred[:, :, -1] - gt[:, -1].unsqueeze(1)).norm(dim=-1)  # (B, M)


def winner_takes_all_loss(pred: torch.Tensor, logits: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``L = Huber(μ_{m*}, τ) + CE(π, m*)``, ``m* = argmin_m ADE_m`` (no gradient through the argmin).

    Returns (loss, m* (B,)).  pred (B, M, T, 2), logits (B, M), gt (B, T, 2).
    """
    with torch.no_grad():
        best = ade_per_mode(pred, gt).argmin(dim=1)  # (B,)
    idx = best.view(-1, 1, 1, 1).expand(-1, 1, pred.shape[2], 2)  # (B, 1, T, 2)
    chosen = pred.gather(1, idx).squeeze(1)  # (B, T, 2) the winning mode's trajectory
    reg = F.smooth_l1_loss(chosen, gt)
    cls = F.cross_entropy(logits, best)
    return reg + cls, best


def min_ade(pred: torch.Tensor, gt: torch.Tensor, k: int | None = None, logits: torch.Tensor | None = None) -> torch.Tensor:
    """minADE over the top-``k`` modes by ``logits`` (all modes if ``k`` is None).  Returns the batch mean."""
    ade = ade_per_mode(pred, gt)  # (B, M)
    if k is not None and logits is not None:
        top = logits.topk(k, dim=1).indices  # (B, k)
        ade = ade.gather(1, top)  # (B, k)
    return ade.min(dim=1).values.mean()


def min_fde(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """minFDE over all modes, batch mean."""
    return fde_per_mode(pred, gt).min(dim=1).values.mean()


def miss_rate(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
    """Fraction of examples whose best mode ends more than ``threshold`` metres from the truth."""
    return (fde_per_mode(pred, gt).min(dim=1).values > threshold).float().mean()
