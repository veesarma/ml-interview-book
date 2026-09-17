"""Position-wise feed-forward networks.

Classic (Vaswani 2017, GPT-2, BERT):   FFN(x) = act(x W_1 + b_1) W_2 + b_2,  d_ff = 4 d_model
Gated (Shazeer 2020; LLaMA, PaLM):     SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down
                                       GeGLU(x)  = (GELU(x W_gate) * (x W_up)) W_down
The gated variants have three matrices; to keep parameter count equal to the classic
2 * d * 4d, LLaMA uses d_ff = 2/3 * 4 d (rounded up to a multiple of 256 in the paper).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """Two-layer MLP applied independently at every position. (B, T, d) -> (B, T, d)."""

    def __init__(self, d_model: int, d_ff: int | None = None, activation: str = "gelu", dropout: float = 0.0) -> None:
        super().__init__()
        d_ff = 4 * d_model if d_ff is None else d_ff
        self.W_1 = nn.Linear(d_model, d_ff)  # (d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)  # (d_ff, d_model)
        self.act = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[activation]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.W_1(x))  # (B, T, d_ff)
        return self.dropout(self.W_2(h))  # (B, T, d_model)


def swiglu_hidden_size(d_model: int, multiple_of: int = 8) -> int:
    """LLaMA's convention: d_ff = ceil(2/3 * 4 d_model / multiple_of) * multiple_of."""
    raw = int(2 * (4 * d_model) / 3)
    return multiple_of * ((raw + multiple_of - 1) // multiple_of)


class GatedFeedForward(nn.Module):
    """SwiGLU (default) or GeGLU. Three projections, no biases (LLaMA style).

    (B, T, d) -> (B, T, d) via  (act(x W_gate) * (x W_up)) W_down.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, activation: str = "silu", multiple_of: int = 8) -> None:
        super().__init__()
        d_ff = swiglu_hidden_size(d_model, multiple_of) if d_ff is None else d_ff
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)  # (d_model, d_ff)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)  # (d_model, d_ff)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)  # (d_ff, d_model)
        self.act = {"silu": F.silu, "gelu": F.gelu}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.W_gate(x))  # (B, T, d_ff)
        up = self.W_up(x)  # (B, T, d_ff)
        return self.W_down(gate * up)  # (B, T, d_ff) -> (B, T, d_model)
