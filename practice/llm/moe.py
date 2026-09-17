# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/moe.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k moe -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/moe --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(n_tokens, d_model) -> (n_tokens, d_model)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TopKRouter(nn.Module):
    """Linear gate + softmax + top-k with renormalised weights."""

    def __init__(self, d_model: int, n_experts: int, k: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route ``x``.

        Args:
            x: (N, d_model) tokens (batch and time already flattened).
        Returns:
            weights: (N, k) gate values, rows sum to 1.
            indices: (N, k) chosen expert ids.
            probs:   (N, E) full softmax, needed by the balance loss.
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def load_balancing_loss(probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Switch-style auxiliary loss ``E Σ_i f_i P_i`` (=1 when perfectly balanced).

    Args:
        probs:   (N, E) router softmax.
        indices: (N, k) selected experts per token.
    Returns:
        scalar loss.
    """
    raise NotImplementedError('TODO: implement load_balancing_loss (see the reference in src/mlbook)')

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

    def __init__(self, d_model: int, d_ff: int, n_experts: int, k: int=2, n_shared: int=0, capacity_factor: float | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: x (B, T, d_model). Returns (y (B, T, d_model), aux_loss scalar)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def expert_usage_fraction(indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """(E,) fraction of routing slots assigned to each expert (sums to 1)."""
    raise NotImplementedError('TODO: implement expert_usage_fraction (see the reference in src/mlbook)')
