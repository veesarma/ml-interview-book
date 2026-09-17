# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/sequence/gru.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gru -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py sequence/gru --force

"""GRU in PyTorch with three explicit gate projections (matches ``torch.nn.GRU`` maths).

Equations (Cho et al., 2014; PyTorch's parameterisation of the candidate):

    r_t = sigmoid(x_t W_xr + h_{t-1} W_hr + b_r)             reset gate
    z_t = sigmoid(x_t W_xz + h_{t-1} W_hz + b_z)             update gate
    n_t = tanh(x_t W_xn + b_xn + r_t * (h_{t-1} W_hn + b_hn)) candidate
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}                    interpolation

Compared with the LSTM: no separate cell state, two gates instead of three,
and the update gate z_t plays both the forget and input roles (they sum to 1).
"""
from __future__ import annotations
import torch
from torch import nn

class GRUCell(nn.Module):
    """One GRU step. Six ``nn.Linear`` layers, one per (gate, source) pair."""

    def __init__(self, d_in: int, d_h: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (B, d_in), h_prev: (B, d_h) -> h_t: (B, d_h)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class GRU(nn.Module):
    """Batch-first single-layer GRU built from ``GRUCell``.

    Input (B, T, d_in) -> outputs (B, T, d_h) and final state (B, d_h).
    """

    def __init__(self, d_in: int, d_h: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential loop over T -- this loop is the reason RNNs do not parallelise across time."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class BidirectionalGRU(nn.Module):
    """Two independent GRUs read the sequence forwards and backwards; outputs are concatenated.

    Input (B, T, d_in) -> (B, T, 2*d_h). Requires the whole sequence up front, so it is
    an encoder-side tool only (never usable for streaming/causal decoding).
    """

    def __init__(self, d_in: int, d_h: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
