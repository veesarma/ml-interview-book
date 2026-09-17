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
        super().__init__()
        self.down = nn.Linear(d_model, r)  # (d -> r)
        self.up = nn.Linear(r, d_model)  # (r -> d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        z = F.gelu(self.down(h))  # (B, T, r)
        return h + self.up(z)  # (B, T, d_model) residual so init == identity


class AdaptedSublayer(nn.Module):
    """Wrap any ``(B, T, d) -> (B, T, d)`` sublayer with a frozen body and a trainable adapter."""

    def __init__(self, sublayer: nn.Module, d_model: int, r: int) -> None:
        super().__init__()
        self.sublayer = sublayer
        for p in self.sublayer.parameters():
            p.requires_grad_(False)
        self.adapter = BottleneckAdapter(d_model, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        return self.adapter(self.sublayer(x))  # (B, T, d_model)
