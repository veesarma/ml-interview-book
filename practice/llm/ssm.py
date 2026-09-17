# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/ssm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ssm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/ssm --force

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
    raise NotImplementedError('TODO: implement discretize_zoh (see the reference in src/mlbook)')

def ssm_recurrent(x: torch.Tensor, A_bar: torch.Tensor, B_bar: torch.Tensor, C: torch.Tensor, D: float) -> torch.Tensor:
    """Run the recurrence h_t = Ā h_{t−1} + B̄ x_t, y_t = C h_t + D x_t.

    Args:
        x: (T,) scalar input sequence.
        A_bar, B_bar, C: (N,) each.
    Returns:
        y: (T,).
    """
    raise NotImplementedError('TODO: implement ssm_recurrent (see the reference in src/mlbook)')

def ssm_kernel(A_bar: torch.Tensor, B_bar: torch.Tensor, C: torch.Tensor, L: int) -> torch.Tensor:
    """Convolution kernel K_l = C Ā^l B̄ for l = 0..L−1, shape (L,)."""
    raise NotImplementedError('TODO: implement ssm_kernel (see the reference in src/mlbook)')

def ssm_convolutional(x: torch.Tensor, K: torch.Tensor, D: float) -> torch.Tensor:
    """Causal convolution y_t = Σ_{l≤t} K_l x_{t−l} + D x_t via one conv1d call.

    Args:
        x: (T,), K: (L,) with L ≥ T.
    Returns:
        y: (T,).
    """
    raise NotImplementedError('TODO: implement ssm_convolutional (see the reference in src/mlbook)')

def selective_scan(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
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
    raise NotImplementedError('TODO: implement selective_scan (see the reference in src/mlbook)')

class SelectiveSSM(nn.Module):
    """Toy S6 block: x -> (Δ, B, C) via three separate linears, then selective scan."""

    def __init__(self, d_model: int, d_state: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, T, d_model)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
