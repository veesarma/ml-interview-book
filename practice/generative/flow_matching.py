# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/flow_matching.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k flow_matching -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/flow_matching --force

"""Conditional flow matching with the linear (rectified-flow) probability path.

Path        x_t = (1 − t) x_0 + t x_1,     x_0 ~ N(0, I) (noise),  x_1 ~ data,  t ~ U[0, 1]
Target      u_t(x_t | x_0, x_1) = x_1 − x_0
Objective   L = E_{t, x_0, x_1} || v_θ(x_t, t) − (x_1 − x_0) ||²
Sampling    integrate dx/dt = v_θ(x, t) from t = 0 to t = 1 (Euler with few steps).

Convention: here t = 0 is *noise* and t = 1 is *data* (flow-matching papers), the opposite of
DDPM's index where t = 0 is data.  ``ddpm.py`` uses the DDPM convention.
"""
from __future__ import annotations
import torch
from torch import nn
from mlbook.generative.ddpm import sinusoidal_embedding

class VelocityMLP(nn.Module):
    """``v_θ(x, t)`` for vectors with continuous t ∈ [0, 1] (scaled to ~[0, 1000] for the embedding).

    forward: x (B, d), t (B,) float -> (B, d).
    """

    def __init__(self, d_x: int, d_hidden: int=128, d_time: int=32) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def linear_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow interpolant and its conditional velocity.

    Args:
        x0 (noise), x1 (data): (B, d).  t: (B,) in [0, 1].
    Returns:
        x_t: (B, d) = (1 − t) x0 + t x1;   u_t: (B, d) = x1 − x0.
    """
    raise NotImplementedError('TODO: implement linear_path (see the reference in src/mlbook)')

def gaussian_path(x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, sigma_min: float=0.001) -> tuple[torch.Tensor, torch.Tensor]:
    """Lipman et al.'s optimal-transport Gaussian path p_t(x | x1) = N(t x1, (1 − (1 − σ_min) t)² I).

    Args:
        x1 (data), x0 (noise): (B, d).  t: (B,).
    Returns:
        x_t: (B, d) = t x1 + (1 − (1 − σ_min) t) x0;   u_t: (B, d) = x1 − (1 − σ_min) x0.
    """
    raise NotImplementedError('TODO: implement gaussian_path (see the reference in src/mlbook)')

def cfm_loss(model: VelocityMLP, x1: torch.Tensor) -> torch.Tensor:
    """Conditional flow-matching loss with the linear path.

    Args:
        x1: (B, d) data batch.
    Returns:
        scalar  E || v_θ(x_t, t) − (x1 − x0) ||².
    """
    raise NotImplementedError('TODO: implement cfm_loss (see the reference in src/mlbook)')

@torch.no_grad()
def sample_euler(model: VelocityMLP, n: int, d: int, n_steps: int, return_trajectory: bool=False) -> torch.Tensor:
    """Integrate dx/dt = v_θ(x, t) from t = 0 (noise) to t = 1 (data) with fixed-step Euler.

    Returns:
        x_1: (N, d), or the trajectory (S+1, N, d) if ``return_trajectory``.
    """
    raise NotImplementedError('TODO: implement sample_euler (see the reference in src/mlbook)')

@torch.no_grad()
def sample_midpoint(model: VelocityMLP, n: int, d: int, n_steps: int) -> torch.Tensor:
    """Second-order (midpoint) integrator; two function evaluations per step.  Returns (N, d)."""
    raise NotImplementedError('TODO: implement sample_midpoint (see the reference in src/mlbook)')

def divergence_exact(model: VelocityMLP, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``tr(∂v/∂x)`` by one backward pass per input dimension (fine for d = 2; use Hutchinson at scale).

    The instantaneous change of variables is  d log p(x(t)) / dt = −tr(∂v/∂x).
    Args:
        x: (B, d).  t: (B,).
    Returns:
        div: (B,).
    """
    raise NotImplementedError('TODO: implement divergence_exact (see the reference in src/mlbook)')

def divergence_hutchinson(model: VelocityMLP, x: torch.Tensor, t: torch.Tensor, n_probes: int=1) -> torch.Tensor:
    """Hutchinson estimator ``E_ε[ε^T (∂v/∂x) ε]``, ε ~ Rademacher: one vector-Jacobian product per probe.

    Args:
        x: (B, d).  t: (B,).
    Returns:
        div_estimate: (B,).
    """
    raise NotImplementedError('TODO: implement divergence_hutchinson (see the reference in src/mlbook)')
