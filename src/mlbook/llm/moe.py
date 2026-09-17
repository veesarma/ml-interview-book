"""A sparse Mixture-of-Experts feed-forward layer with top-k routing and the
Switch/GShard load-balancing auxiliary loss (PyTorch).

    y = Σ_{i ∈ TopK(x)} g_i(x) E_i(x)   (+ Σ_shared E_s(x) in DeepSeek-style MoE)

Routing: ``p = softmax(x W_g)`` over ``E`` experts, keep the top ``k``, renormalise
their probabilities to sum to 1, dispatch each token to its ``k`` experts.

Balance loss (Switch Transformer, eq. 4; GShard is the same idea):

    L_aux = E · Σ_i f_i · P_i,   f_i = fraction of tokens routed to expert i,
                                 P_i = mean router probability of expert i.

At perfect balance f_i = P_i = 1/E and L_aux = 1; imbalance raises it.  Only the
P_i term is differentiable; f_i is a constant that says *which* expert's
probability to push down.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """One expert: a SwiGLU-free, plain two-layer MLP ``x -> W2 silu(W1 x)``."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n_tokens, d_model) -> (n_tokens, d_model)."""
        h = F.silu(self.w1(x))  # (n_tokens, d_ff)
        return self.w2(h)  # (n_tokens, d_model)


class TopKRouter(nn.Module):
    """Linear gate + softmax + top-k with renormalised weights."""

    def __init__(self, d_model: int, n_experts: int, k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.k = k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route ``x``.

        Args:
            x: (N, d_model) tokens (batch and time already flattened).
        Returns:
            weights: (N, k) gate values, rows sum to 1.
            indices: (N, k) chosen expert ids.
            probs:   (N, E) full softmax, needed by the balance loss.
        """
        logits = self.gate(x)  # (N, E)
        probs = torch.softmax(logits, dim=-1)  # (N, E)
        top_p, indices = probs.topk(self.k, dim=-1)  # (N, k), (N, k)
        weights = top_p / top_p.sum(dim=-1, keepdim=True)  # (N, k)
        return weights, indices, probs


def load_balancing_loss(probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Switch-style auxiliary loss ``E Σ_i f_i P_i`` (=1 when perfectly balanced).

    Args:
        probs:   (N, E) router softmax.
        indices: (N, k) selected experts per token.
    Returns:
        scalar loss.
    """
    N, E = probs.shape
    k = indices.shape[1]
    one_hot = F.one_hot(indices, E).sum(dim=1).float()  # (N, E) 1 where expert chosen
    f = one_hot.sum(dim=0) / (N * k)  # (E,) dispatch fraction, sums to 1
    P = probs.mean(dim=0)  # (E,) mean gate probability, sums to 1
    return E * torch.sum(f * P)


class MoELayer(nn.Module):
    """Sparse MoE FFN with optional expert capacity (token dropping) and shared experts.

    Args:
        n_experts:      routed experts ``E``.
        k:              experts per token.
        n_shared:       always-on experts added to every token (DeepSeek-V2/V3 style).
        capacity_factor: if set, each expert accepts at most
            ``ceil(capacity_factor * N * k / E)`` tokens per forward; extra tokens
            are dropped (their expert contribution is zero, the residual stream
            still carries them).  ``None`` = no dropping.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int,
        k: int = 2,
        n_shared: int = 0,
        capacity_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.router = TopKRouter(d_model, n_experts, k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])
        self.shared = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_shared)])
        self.n_experts, self.k, self.capacity_factor = n_experts, k, capacity_factor
        self.last_usage: torch.Tensor | None = None  # (E,) tokens processed per expert

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: x (B, T, d_model). Returns (y (B, T, d_model), aux_loss scalar)."""
        B, T, d = x.shape
        flat = x.reshape(B * T, d)  # (N, d_model), N = B*T
        weights, indices, probs = self.router(flat)  # (N, k), (N, k), (N, E)
        y = torch.zeros_like(flat)  # (N, d_model)
        capacity = None
        if self.capacity_factor is not None:
            capacity = int(-(-self.capacity_factor * flat.shape[0] * self.k // self.n_experts))
        usage = torch.zeros(self.n_experts)
        for e, expert in enumerate(self.experts):
            token_idx, slot = torch.where(indices == e)  # (n_e,), (n_e,) which tokens chose e
            if capacity is not None and token_idx.numel() > capacity:
                token_idx, slot = token_idx[:capacity], slot[:capacity]  # drop the overflow
            if token_idx.numel() == 0:
                continue
            usage[e] = token_idx.numel()
            out = expert(flat[token_idx])  # (n_e, d_model)
            y.index_add_(0, token_idx, weights[token_idx, slot, None] * out)  # scatter-add
        for expert in self.shared:
            y = y + expert(flat)  # (N, d_model) shared experts see every token
        self.last_usage = usage
        return y.reshape(B, T, d), load_balancing_loss(probs, indices)


def expert_usage_fraction(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """(E,) fraction of routing slots assigned to each expert (sums to 1)."""
    counts = torch.bincount(indices.reshape(-1), minlength=n_experts).float()  # (E,)
    return counts / counts.sum()
