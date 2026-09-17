# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/ffn.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ffn -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/ffn --force

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

    def __init__(self, d_model: int, d_ff: int | None=None, activation: str='gelu', dropout: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def swiglu_hidden_size(d_model: int, multiple_of: int=8) -> int:
    """LLaMA's convention: d_ff = ceil(2/3 * 4 d_model / multiple_of) * multiple_of."""
    raise NotImplementedError('TODO: implement swiglu_hidden_size (see the reference in src/mlbook)')

class GatedFeedForward(nn.Module):
    """SwiGLU (default) or GeGLU. Three projections, no biases (LLaMA style).

    (B, T, d) -> (B, T, d) via  (act(x W_gate) * (x W_up)) W_down.
    """

    def __init__(self, d_model: int, d_ff: int | None=None, activation: str='silu', multiple_of: int=8) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
