# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/generative/ddpm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ddpm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py generative/ddpm --force

"""Denoising diffusion (DDPM / DDIM) on vectors, with eps-/x0-/v-parameterisations and
classifier-free guidance.

Forward process   q(x_t | x_{t-1}) = N( sqrt(1 − β_t) x_{t-1}, β_t I )
Closed form       q(x_t | x_0)     = N( sqrt(ᾱ_t) x_0, (1 − ᾱ_t) I ),   ᾱ_t = Π_{s≤t} (1 − β_s)
Training          L = || ε − ε_θ( sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε, t ) ||²
Posterior         q(x_{t-1} | x_t, x_0) = N( μ̃_t, β̃_t I ),
                  μ̃_t = [ sqrt(ᾱ_{t-1}) β_t x_0 + sqrt(α_t)(1 − ᾱ_{t-1}) x_t ] / (1 − ᾱ_t),
                  β̃_t = (1 − ᾱ_{t-1}) β_t / (1 − ᾱ_t).
Timesteps are integers ``t ∈ {0, …, T−1}`` indexing ``betas``; ``t = 0`` is the least noisy step.
"""
from __future__ import annotations
import math
import torch
from torch import nn

def linear_beta_schedule(T: int, beta_start: float=0.0001, beta_end: float=0.02) -> torch.Tensor:
    """DDPM's linear schedule.  Returns betas (T,)."""
    raise NotImplementedError('TODO: implement linear_beta_schedule (see the reference in src/mlbook)')

def cosine_alpha_bar_schedule(T: int, s: float=0.008, max_beta: float=0.999) -> torch.Tensor:
    """Improved-DDPM cosine schedule defined through ᾱ(t) = cos²(((t/T + s)/(1 + s)) · π/2).

    Returns:
        betas (T,) with β_t = 1 − ᾱ_t / ᾱ_{t−1}, clipped at ``max_beta``.
    """
    raise NotImplementedError('TODO: implement cosine_alpha_bar_schedule (see the reference in src/mlbook)')

def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Rescale a schedule so that ᾱ_T = 0 exactly (SNR(T) = 0), keeping ᾱ_1 fixed.

    Operates on sqrt(ᾱ): shift so the last value is 0, then scale so the first is unchanged.
    Returns:
        betas (T,).
    """
    raise NotImplementedError('TODO: implement enforce_zero_terminal_snr (see the reference in src/mlbook)')

class NoiseSchedule:
    """Precomputed schedule constants.  All tensors are (T,)."""

    def __init__(self, betas: torch.Tensor) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def snr(self) -> torch.Tensor:
        """Signal-to-noise ratio ᾱ_t / (1 − ᾱ_t), (T,)."""
        raise NotImplementedError('TODO: implement snr (see the reference in src/mlbook)')

def gather(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Pick ``values[t]`` and reshape to broadcast against a batch of ``ndim``-dim tensors.

    Args:
        values: (T,).  t: (B,) integer timesteps.
    Returns:
        (B, 1, ..., 1) with ``ndim`` dimensions total.
    """
    raise NotImplementedError('TODO: implement gather (see the reference in src/mlbook)')

def q_sample(sched: NoiseSchedule, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """``x_t = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε``.

    Args:
        x0, eps: (B, d).  t: (B,).
    Returns:
        x_t: (B, d).
    """
    raise NotImplementedError('TODO: implement q_sample (see the reference in src/mlbook)')

def eps_to_x0(sched: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Invert q_sample: ``x_0 = (x_t − sqrt(1 − ᾱ_t) ε) / sqrt(ᾱ_t)``.  Shapes as q_sample."""
    raise NotImplementedError('TODO: implement eps_to_x0 (see the reference in src/mlbook)')

def x0_eps_to_v(sched: NoiseSchedule, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v-parameterisation target ``v = sqrt(ᾱ_t) ε − sqrt(1 − ᾱ_t) x_0``."""
    raise NotImplementedError('TODO: implement x0_eps_to_v (see the reference in src/mlbook)')

def v_to_eps_x0(sched: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """From a v prediction recover ``x_0 = sqrt(ᾱ_t) x_t − sqrt(1 − ᾱ_t) v`` and ``ε = sqrt(1 − ᾱ_t) x_t + sqrt(ᾱ_t) v``."""
    raise NotImplementedError('TODO: implement v_to_eps_x0 (see the reference in src/mlbook)')

def posterior_mean_variance(sched: NoiseSchedule, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q(x_{t−1} | x_t, x_0)`` mean μ̃_t and variance β̃_t.

    Args:
        x_t, x0: (B, d).  t: (B,).
    Returns:
        (mean (B, d), variance (B, 1)).
    """
    raise NotImplementedError('TODO: implement posterior_mean_variance (see the reference in src/mlbook)')

def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float=10000.0) -> torch.Tensor:
    """Transformer-style embedding of integer timesteps.

    Args:
        t: (B,) integers.
    Returns:
        (B, dim).
    """
    raise NotImplementedError('TODO: implement sinusoidal_embedding (see the reference in src/mlbook)')

class EpsMLP(nn.Module):
    """``ε_θ(x_t, t, y)`` for vectors.  ``y = n_classes`` is the null (unconditional) token.

    forward: x_t (B, d), t (B,), y (B,) or None -> (B, d).
    """

    def __init__(self, d_x: int, n_classes: int=0, d_hidden: int=128, d_time: int=32) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None=None) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def ddpm_loss(model: EpsMLP, sched: NoiseSchedule, x0: torch.Tensor, y: torch.Tensor | None=None, p_uncond: float=0.0) -> torch.Tensor:
    """Simplified DDPM objective ``E_{t, ε} || ε − ε_θ(x_t, t, y) ||²`` with label dropout for CFG.

    Args:
        x0: (B, d).  y: (B,) labels or None.  p_uncond: probability of replacing y by the null token.
    Returns:
        scalar.
    """
    raise NotImplementedError('TODO: implement ddpm_loss (see the reference in src/mlbook)')

def predict_eps_cfg(model: EpsMLP, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None, guidance_scale: float) -> torch.Tensor:
    """Classifier-free guidance ``ε̃ = (1 + w) ε_θ(x_t, y) − w ε_θ(x_t, ∅)``  (w = 0: plain conditional).

    Args:
        x_t: (B, d).  t: (B,).  y: (B,) or None.
    Returns:
        ε̃: (B, d).
    """
    raise NotImplementedError('TODO: implement predict_eps_cfg (see the reference in src/mlbook)')

@torch.no_grad()
def sample_ancestral(model: EpsMLP, sched: NoiseSchedule, n: int, d: int, y: torch.Tensor | None=None, guidance_scale: float=0.0, clip_x0: float | None=None) -> torch.Tensor:
    """DDPM sampling: for t = T−1 … 0,  x_{t−1} = μ̃_t(x_t, x̂_0) + sqrt(β̃_t) z.

    ``clip_x0`` clamps the implied x̂_0 to ``[−clip_x0, clip_x0]`` at every step.  At large t,
    ᾱ_t is tiny and x̂_0 = (x_t − sqrt(1−ᾱ_t) ε̂)/sqrt(ᾱ_t) divides by a number near zero, so a
    small error in ε̂ becomes a huge x̂_0.  Image models clamp to [−1, 1] for exactly this reason.

    Returns:
        x_0: (N, d).
    """
    raise NotImplementedError('TODO: implement sample_ancestral (see the reference in src/mlbook)')

@torch.no_grad()
def sample_ddim(model: EpsMLP, sched: NoiseSchedule, n: int, d: int, n_steps: int, eta: float=0.0, y: torch.Tensor | None=None, guidance_scale: float=0.0, clip_x0: float | None=None, return_trajectory: bool=False) -> torch.Tensor:
    """DDIM sampling on a sub-sequence of ``n_steps`` timesteps (η = 0 is deterministic).

    x_{τ_{i−1}} = sqrt(ᾱ_{τ_{i−1}}) x̂_0 + sqrt(1 − ᾱ_{τ_{i−1}} − σ²) ε̂ + σ z,
    σ = η sqrt((1 − ᾱ_prev)/(1 − ᾱ)) sqrt(1 − ᾱ/ᾱ_prev).

    Returns:
        x_0: (N, d).
    """
    raise NotImplementedError('TODO: implement sample_ddim (see the reference in src/mlbook)')

def score_from_eps(sched: NoiseSchedule, eps_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``∇_x log p_t(x_t) ≈ −ε_θ / sqrt(1 − ᾱ_t)`` (Tweedie / denoising score matching).  Shapes (B, d)."""
    raise NotImplementedError('TODO: implement score_from_eps (see the reference in src/mlbook)')
