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
        super().__init__()
        self.d_in, self.d_h = d_in, d_h
        self.x_r = nn.Linear(d_in, d_h)  # W_xr, b_r(x part)
        self.h_r = nn.Linear(d_h, d_h)  # W_hr
        self.x_z = nn.Linear(d_in, d_h)  # W_xz
        self.h_z = nn.Linear(d_h, d_h)  # W_hz
        self.x_n = nn.Linear(d_in, d_h)  # W_xn, b_xn
        self.h_n = nn.Linear(d_h, d_h)  # W_hn, b_hn

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (B, d_in), h_prev: (B, d_h) -> h_t: (B, d_h)."""
        r = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev))  # (B, d_h)
        z = torch.sigmoid(self.x_z(x_t) + self.h_z(h_prev))  # (B, d_h)
        n = torch.tanh(self.x_n(x_t) + r * self.h_n(h_prev))  # (B, d_h)
        h_t = (1.0 - z) * n + z * h_prev  # (B, d_h)
        return h_t


class GRU(nn.Module):
    """Batch-first single-layer GRU built from ``GRUCell``.

    Input (B, T, d_in) -> outputs (B, T, d_h) and final state (B, d_h).
    """

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.cell = GRUCell(d_in, d_h)
        self.d_h = d_h

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequential loop over T -- this loop is the reason RNNs do not parallelise across time."""
        B, T, _ = x.shape
        h = x.new_zeros(B, self.d_h) if h0 is None else h0  # (B, d_h)
        outputs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)  # (B, d_h); depends on h from step t-1
            outputs.append(h)
        H = torch.stack(outputs, dim=1)  # (B, T, d_h)
        return H, h


class BidirectionalGRU(nn.Module):
    """Two independent GRUs read the sequence forwards and backwards; outputs are concatenated.

    Input (B, T, d_in) -> (B, T, 2*d_h). Requires the whole sequence up front, so it is
    an encoder-side tool only (never usable for streaming/causal decoding).
    """

    def __init__(self, d_in: int, d_h: int) -> None:
        super().__init__()
        self.fwd = GRU(d_in, d_h)
        self.bwd = GRU(d_in, d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H_f, _ = self.fwd(x)  # (B, T, d_h) left-to-right
        H_b, _ = self.bwd(torch.flip(x, dims=[1]))  # (B, T, d_h) computed on the reversed sequence
        H_b = torch.flip(H_b, dims=[1])  # (B, T, d_h) re-aligned so row t is "the future of t"
        return torch.cat([H_f, H_b], dim=-1)  # (B, T, 2*d_h)
