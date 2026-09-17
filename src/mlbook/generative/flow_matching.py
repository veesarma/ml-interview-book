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

    def __init__(self, d_x: int, d_hidden: int = 128, d_time: int = 32) -> None:
        super().__init__()
        self.d_time = d_time
        self.time_mlp = nn.Sequential(nn.Linear(d_time, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.in_proj = nn.Linear(d_x, d_hidden)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.block2 = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_x))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_embedding(t * 1000.0, self.d_time))      # (B, d_hidden)
        h = self.in_proj(x) + temb                                                # (B, d_hidden)
        h = h + self.block1(h)                                                    # (B, d_hidden)
        h = h + self.block2(h)                                                    # (B, d_hidden)
        return self.out(h)                                                        # (B, d)


def linear_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectified-flow interpolant and its conditional velocity.

    Args:
        x0 (noise), x1 (data): (B, d).  t: (B,) in [0, 1].
    Returns:
        x_t: (B, d) = (1 − t) x0 + t x1;   u_t: (B, d) = x1 − x0.
    """
    tt = t[:, None]                                                               # (B, 1)
    x_t = (1.0 - tt) * x0 + tt * x1                                               # (B, d)
    u_t = x1 - x0                                                                 # (B, d)
    return x_t, u_t


def gaussian_path(x1: torch.Tensor, x0: torch.Tensor, t: torch.Tensor, sigma_min: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Lipman et al.'s optimal-transport Gaussian path p_t(x | x1) = N(t x1, (1 − (1 − σ_min) t)² I).

    Args:
        x1 (data), x0 (noise): (B, d).  t: (B,).
    Returns:
        x_t: (B, d) = t x1 + (1 − (1 − σ_min) t) x0;   u_t: (B, d) = x1 − (1 − σ_min) x0.
    """
    tt = t[:, None]                                                               # (B, 1)
    x_t = tt * x1 + (1.0 - (1.0 - sigma_min) * tt) * x0                           # (B, d)
    u_t = x1 - (1.0 - sigma_min) * x0                                             # (B, d)
    return x_t, u_t


def cfm_loss(model: VelocityMLP, x1: torch.Tensor) -> torch.Tensor:
    """Conditional flow-matching loss with the linear path.

    Args:
        x1: (B, d) data batch.
    Returns:
        scalar  E || v_θ(x_t, t) − (x1 − x0) ||².
    """
    B = x1.shape[0]
    x0 = torch.randn_like(x1)                                                     # (B, d)
    t = torch.rand(B)                                                             # (B,)
    x_t, u_t = linear_path(x0, x1, t)                                             # (B, d), (B, d)
    v = model(x_t, t)                                                             # (B, d)
    return ((v - u_t) ** 2).mean()


@torch.no_grad()
def sample_euler(model: VelocityMLP, n: int, d: int, n_steps: int, return_trajectory: bool = False) -> torch.Tensor:
    """Integrate dx/dt = v_θ(x, t) from t = 0 (noise) to t = 1 (data) with fixed-step Euler.

    Returns:
        x_1: (N, d), or the trajectory (S+1, N, d) if ``return_trajectory``.
    """
    x = torch.randn(n, d)                                                         # (N, d)
    traj = [x]
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i * dt)                                              # (N,)
        x = x + dt * model(x, t)                                                  # (N, d)
        traj.append(x)
    return torch.stack(traj, dim=0) if return_trajectory else x                  # (S+1, N, d) or (N, d)


@torch.no_grad()
def sample_midpoint(model: VelocityMLP, n: int, d: int, n_steps: int) -> torch.Tensor:
    """Second-order (midpoint) integrator; two function evaluations per step.  Returns (N, d)."""
    x = torch.randn(n, d)                                                         # (N, d)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n,), i * dt)                                              # (N,)
        v_half = model(x + 0.5 * dt * model(x, t), t + 0.5 * dt)                  # (N, d)
        x = x + dt * v_half                                                       # (N, d)
    return x


def divergence_exact(model: VelocityMLP, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``tr(∂v/∂x)`` by one backward pass per input dimension (fine for d = 2; use Hutchinson at scale).

    The instantaneous change of variables is  d log p(x(t)) / dt = −tr(∂v/∂x).
    Args:
        x: (B, d).  t: (B,).
    Returns:
        div: (B,).
    """
    x = x.detach().requires_grad_(True)
    v = model(x, t)                                                               # (B, d)
    div = torch.zeros(x.shape[0])                                                 # (B,)
    for j in range(x.shape[1]):
        grad_j = torch.autograd.grad(v[:, j].sum(), x, retain_graph=True)[0]      # (B, d) = ∂v_j/∂x
        div = div + grad_j[:, j]                                                  # accumulate ∂v_j/∂x_j
    return div


def divergence_hutchinson(model: VelocityMLP, x: torch.Tensor, t: torch.Tensor, n_probes: int = 1) -> torch.Tensor:
    """Hutchinson estimator ``E_ε[ε^T (∂v/∂x) ε]``, ε ~ Rademacher: one vector-Jacobian product per probe.

    Args:
        x: (B, d).  t: (B,).
    Returns:
        div_estimate: (B,).
    """
    x = x.detach().requires_grad_(True)
    v = model(x, t)                                                               # (B, d)
    est = torch.zeros(x.shape[0])                                                 # (B,)
    for _ in range(n_probes):
        eps = torch.randint(0, 2, x.shape).float() * 2.0 - 1.0                    # (B, d) ±1
        vjp = torch.autograd.grad(v, x, grad_outputs=eps, retain_graph=True)[0]   # (B, d) = ε^T J
        est = est + (vjp * eps).sum(dim=1)                                        # (B,)
    return est / n_probes
