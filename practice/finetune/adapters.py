# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/finetune/adapters.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k adapters -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py finetune/adapters --force

"""Bottleneck adapters (Houlsby et al. 2019): small residual MLPs inserted after a sublayer.

    h' = h + W_up · σ(W_down · h),   W_down ∈ R^{r×d}, W_up ∈ R^{d×r}, r ≪ d

``W_up`` starts at zero so the adapter is the identity at initialisation.  Unlike
LoRA, an adapter adds sequential compute at inference and cannot be merged away.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckAdapter(nn.Module):
    """Residual down-project / nonlinearity / up-project block."""

    def __init__(self, d_model: int, r: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class AdaptedSublayer(nn.Module):
    """Wrap any ``(B, T, d) -> (B, T, d)`` sublayer with a frozen body and a trainable adapter."""

    def __init__(self, sublayer: nn.Module, d_model: int, r: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
