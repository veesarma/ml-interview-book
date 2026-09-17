"""State-space models from the continuous system to Mamba's selective scan (PyTorch).

Continuous:   h'(t) = A h(t) + B x(t),   y(t) = C h(t) + D x(t)
ZOH discretisation with step Δ (diagonal A of shape (N,)):
    Ā = exp(Δ A),   B̄ = (Ā − 1) / A · B          (exact for a zero-order-held input)
Recurrence:   h_t = Ā h_{t−1} + B̄ x_t,    y_t = C h_t + D x_t
Convolution:  y = K * x  with  K_l = C Ā^l B̄  (l = 0..L−1), valid because the
              system is linear and time-invariant (LTI): unroll the recurrence.
Selective (Mamba/S6): Δ, B, C become functions of x_t, so the system is no longer
time-invariant: the convolutional shortcut disappears and you must scan.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def discretize_zoh(A: torch.Tensor, B: torch.Tensor, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-order-hold discretisation of a diagonal SSM.

    Args:
        A: (N,) diagonal of the (negative) state matrix.
        B: (N,) input vector.
        delta: step size Δ > 0.
    Returns:
        (A_bar (N,), B_bar (N,)).
    """
    A_bar = torch.exp(delta * A)  # (N,)
    B_bar = (A_bar - 1.0) / A * B  # (N,)  = ∫_0^Δ exp(sA) ds · B
    return A_bar, B_bar


def ssm_recurrent(x: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor, C: torch.Tensor, D: float) -> torch.Tensor:
    """Run the recurrence h_t = Ā h_{t−1} + B̄ x_t, y_t = C h_t + D x_t.

    Args:
        x: (T,) scalar input sequence.
        A_bar, B_bar, C: (N,) each.
    Returns:
        y: (T,).
    """
    T = x.shape[0]
    h = torch.zeros_like(A_bar)  # (N,) hidden state
    y = torch.zeros(T, dtype=x.dtype)  # (T,)
    for t in range(T):
        h = A_bar * h + B_bar * x[t]  # (N,)
        y[t] = torch.dot(C, h) + D * x[t]  # scalar
    return y


def ssm_kernel(A_bar: torch.Tensor, B_bar: torch.Tensor, C: torch.Tensor, L: int) -> torch.Tensor:
    """Convolution kernel K_l = C Ā^l B̄ for l = 0..L−1, shape (L,)."""
    powers = A_bar[None, :] ** torch.arange(L, dtype=A_bar.dtype)[:, None]  # (L, N) = Ā^l
    return powers @ (C * B_bar)  # (L,)  Σ_n C_n Ā_n^l B̄_n


def ssm_convolutional(x: torch.Tensor, K: torch.Tensor, D: float) -> torch.Tensor:
    """Causal convolution y_t = Σ_{l≤t} K_l x_{t−l} + D x_t via one conv1d call.

    Args:
        x: (T,), K: (L,) with L ≥ T.
    Returns:
        y: (T,).
    """
    T = x.shape[0]
    x_pad = F.pad(x, (T - 1, 0))[None, None, :]  # (1, 1, 2T-1) left-pad for causality
    kernel = K[:T].flip(0)[None, None, :]  # (1, 1, T) conv1d cross-correlates, so flip
    y = F.conv1d(x_pad, kernel)[0, 0]  # (T,)
    return y + D * x


def selective_scan(
    x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
) -> torch.Tensor:
    """Mamba's S6 scan with input-dependent Δ, B, C (sequential reference version).

    Per channel d and state n:  Ā_{t,d,n} = exp(Δ_{t,d} A_{d,n}),  B̄_{t,d,n} = Δ_{t,d} B_{t,n}
    (Mamba uses the Euler-style B̄ = Δ B),  h_t = Ā_t ⊙ h_{t−1} + B̄_t x_t,  y_t = ⟨C_t, h_t⟩ + D x_t.

    Args:
        x:     (B, T, d) input.
        delta: (B, T, d) positive step sizes.
        A:     (d, N) state matrix diagonal per channel (negative).
        B, C:  (B, T, N) input-dependent projections.
        D:     (d,) skip.
    Returns:
        y: (B, T, d).
    """
    Bsz, T, d = x.shape
    N = A.shape[1]
    h = torch.zeros(Bsz, d, N, dtype=x.dtype)  # (B, d, N)
    ys = []
    for t in range(T):
        A_bar = torch.exp(delta[:, t, :, None] * A[None])  # (B, d, N)
        B_bar = delta[:, t, :, None] * B[:, t, None, :]  # (B, d, N)
        h = A_bar * h + B_bar * x[:, t, :, None]  # (B, d, N)
        y_t = (h * C[:, t, None, :]).sum(-1) + D[None] * x[:, t]  # (B, d)
        ys.append(y_t)
    return torch.stack(ys, dim=1)  # (B, T, d)


class SelectiveSSM(nn.Module):
    """Toy S6 block: x -> (Δ, B, C) via three separate linears, then selective scan."""

    def __init__(self, d_model: int, d_state: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1))  # (d, N)
        self.D = nn.Parameter(torch.ones(d_model))  # (d,)
        self.to_delta = nn.Linear(d_model, d_model)
        self.to_B = nn.Linear(d_model, d_state, bias=False)
        self.to_C = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        delta = F.softplus(self.to_delta(x))  # (B, T, d) positive step per channel
        B = self.to_B(x)  # (B, T, N)
        C = self.to_C(x)  # (B, T, N)
        A = -torch.exp(self.A_log)  # (d, N) negative for stability
        return selective_scan(x, delta, A, B, C, self.D)  # (B, T, d)
