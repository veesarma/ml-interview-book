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


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """DDPM's linear schedule.  Returns betas (T,)."""
    return torch.linspace(beta_start, beta_end, T)


def cosine_alpha_bar_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Improved-DDPM cosine schedule defined through ᾱ(t) = cos²(((t/T + s)/(1 + s)) · π/2).

    Returns:
        betas (T,) with β_t = 1 − ᾱ_t / ᾱ_{t−1}, clipped at ``max_beta``.
    """
    steps = torch.arange(T + 1, dtype=torch.float64) / T                          # (T+1,)
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2                       # (T+1,)
    alpha_bar = f / f[0]                                                          # (T+1,)  ᾱ_0 = 1
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]                                  # (T,)
    return betas.clamp(max=max_beta).float()


def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """Rescale a schedule so that ᾱ_T = 0 exactly (SNR(T) = 0), keeping ᾱ_1 fixed.

    Operates on sqrt(ᾱ): shift so the last value is 0, then scale so the first is unchanged.
    Returns:
        betas (T,).
    """
    alpha_bar = torch.cumprod(1.0 - betas.double(), dim=0)                        # (T,)
    sqrt_ab = alpha_bar.sqrt()                                                    # (T,)
    first, last = sqrt_ab[0].clone(), sqrt_ab[-1].clone()
    sqrt_ab = sqrt_ab - last                                                      # (T,) now ends at 0
    sqrt_ab = sqrt_ab * first / (first - last)                                    # (T,) starts at original
    alpha_bar = sqrt_ab ** 2                                                      # (T,)
    alphas = alpha_bar / torch.cat([torch.ones(1, dtype=alpha_bar.dtype), alpha_bar[:-1]])  # (T,)
    return (1.0 - alphas).float()


class NoiseSchedule:
    """Precomputed schedule constants.  All tensors are (T,)."""

    def __init__(self, betas: torch.Tensor) -> None:
        self.T = betas.shape[0]
        self.betas = betas                                                        # (T,)
        self.alphas = 1.0 - betas                                                 # (T,)
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)                        # (T,)
        self.alpha_bar_prev = torch.cat([torch.ones(1), self.alpha_bar[:-1]])     # (T,)  ᾱ_{t−1}, ᾱ_{−1} := 1
        self.sqrt_alpha_bar = self.alpha_bar.sqrt()                               # (T,)
        self.sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bar).sqrt()             # (T,)
        self.posterior_variance = (1.0 - self.alpha_bar_prev) * betas / (1.0 - self.alpha_bar)  # (T,)  β̃_t

    def snr(self) -> torch.Tensor:
        """Signal-to-noise ratio ᾱ_t / (1 − ᾱ_t), (T,)."""
        return self.alpha_bar / (1.0 - self.alpha_bar)


def gather(values: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Pick ``values[t]`` and reshape to broadcast against a batch of ``ndim``-dim tensors.

    Args:
        values: (T,).  t: (B,) integer timesteps.
    Returns:
        (B, 1, ..., 1) with ``ndim`` dimensions total.
    """
    out = values[t]                                                               # (B,)
    return out.reshape(-1, *([1] * (ndim - 1)))                                   # (B, 1, ...)


# ---------------------------------------------------------------------------
# Forward process and parameterisation conversions
# ---------------------------------------------------------------------------


def q_sample(sched: NoiseSchedule, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """``x_t = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε``.

    Args:
        x0, eps: (B, d).  t: (B,).
    Returns:
        x_t: (B, d).
    """
    a = gather(sched.sqrt_alpha_bar, t, x0.ndim)                                  # (B, 1)
    s = gather(sched.sqrt_one_minus_alpha_bar, t, x0.ndim)                        # (B, 1)
    return a * x0 + s * eps


def eps_to_x0(sched: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Invert q_sample: ``x_0 = (x_t − sqrt(1 − ᾱ_t) ε) / sqrt(ᾱ_t)``.  Shapes as q_sample."""
    a = gather(sched.sqrt_alpha_bar, t, x_t.ndim)                                 # (B, 1)
    s = gather(sched.sqrt_one_minus_alpha_bar, t, x_t.ndim)                       # (B, 1)
    return (x_t - s * eps) / a


def x0_eps_to_v(sched: NoiseSchedule, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v-parameterisation target ``v = sqrt(ᾱ_t) ε − sqrt(1 − ᾱ_t) x_0``."""
    a = gather(sched.sqrt_alpha_bar, t, x0.ndim)                                  # (B, 1)
    s = gather(sched.sqrt_one_minus_alpha_bar, t, x0.ndim)                        # (B, 1)
    return a * eps - s * x0


def v_to_eps_x0(sched: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """From a v prediction recover ``x_0 = sqrt(ᾱ_t) x_t − sqrt(1 − ᾱ_t) v`` and ``ε = sqrt(1 − ᾱ_t) x_t + sqrt(ᾱ_t) v``."""
    a = gather(sched.sqrt_alpha_bar, t, x_t.ndim)                                 # (B, 1)
    s = gather(sched.sqrt_one_minus_alpha_bar, t, x_t.ndim)                       # (B, 1)
    x0 = a * x_t - s * v                                                          # (B, d)
    eps = s * x_t + a * v                                                         # (B, d)
    return eps, x0


def posterior_mean_variance(sched: NoiseSchedule, x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q(x_{t−1} | x_t, x_0)`` mean μ̃_t and variance β̃_t.

    Args:
        x_t, x0: (B, d).  t: (B,).
    Returns:
        (mean (B, d), variance (B, 1)).
    """
    ab = gather(sched.alpha_bar, t, x_t.ndim)                                     # (B, 1)
    ab_prev = gather(sched.alpha_bar_prev, t, x_t.ndim)                           # (B, 1)
    beta = gather(sched.betas, t, x_t.ndim)                                       # (B, 1)
    alpha = gather(sched.alphas, t, x_t.ndim)                                     # (B, 1)
    coef_x0 = ab_prev.sqrt() * beta / (1.0 - ab)                                  # (B, 1)
    coef_xt = alpha.sqrt() * (1.0 - ab_prev) / (1.0 - ab)                         # (B, 1)
    mean = coef_x0 * x0 + coef_xt * x_t                                           # (B, d)
    var = gather(sched.posterior_variance, t, x_t.ndim)                           # (B, 1)
    return mean, var


# ---------------------------------------------------------------------------
# The noise-prediction network (MLP with sinusoidal time embedding, optional class label)
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Transformer-style embedding of integer timesteps.

    Args:
        t: (B,) integers.
    Returns:
        (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)  # (dim/2,)
    args = t.float()[:, None] * freqs[None, :]                                   # (B, dim/2)
    return torch.cat([args.sin(), args.cos()], dim=1)                            # (B, dim)


class EpsMLP(nn.Module):
    """``ε_θ(x_t, t, y)`` for vectors.  ``y = n_classes`` is the null (unconditional) token.

    forward: x_t (B, d), t (B,), y (B,) or None -> (B, d).
    """

    def __init__(self, d_x: int, n_classes: int = 0, d_hidden: int = 128, d_time: int = 32) -> None:
        super().__init__()
        self.d_time = d_time
        self.n_classes = n_classes
        self.time_mlp = nn.Sequential(nn.Linear(d_time, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.class_emb = nn.Embedding(n_classes + 1, d_hidden) if n_classes > 0 else None  # +1 null token
        self.in_proj = nn.Linear(d_x, d_hidden)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.block2 = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_hidden))
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(d_hidden, d_x))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        cond = self.time_mlp(sinusoidal_embedding(t, self.d_time))               # (B, d_hidden)
        if self.class_emb is not None:
            if y is None:
                y = torch.full((x_t.shape[0],), self.n_classes, dtype=torch.long)  # (B,) all null
            cond = cond + self.class_emb(y)                                       # (B, d_hidden)
        h = self.in_proj(x_t) + cond                                              # (B, d_hidden)
        h = h + self.block1(h)                                                    # (B, d_hidden)
        h = h + self.block2(h)                                                    # (B, d_hidden)
        return self.out(h)                                                        # (B, d)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def ddpm_loss(model: EpsMLP, sched: NoiseSchedule, x0: torch.Tensor, y: torch.Tensor | None = None,
              p_uncond: float = 0.0) -> torch.Tensor:
    """Simplified DDPM objective ``E_{t, ε} || ε − ε_θ(x_t, t, y) ||²`` with label dropout for CFG.

    Args:
        x0: (B, d).  y: (B,) labels or None.  p_uncond: probability of replacing y by the null token.
    Returns:
        scalar.
    """
    B = x0.shape[0]
    t = torch.randint(0, sched.T, (B,))                                           # (B,)
    eps = torch.randn_like(x0)                                                    # (B, d)
    x_t = q_sample(sched, x0, t, eps)                                             # (B, d)
    if y is not None and p_uncond > 0.0:
        drop = torch.rand(B) < p_uncond                                           # (B,) bool
        y = torch.where(drop, torch.full_like(y, model.n_classes), y)             # (B,)
    eps_hat = model(x_t, t, y)                                                    # (B, d)
    return ((eps - eps_hat) ** 2).mean()


# ---------------------------------------------------------------------------
# Sampling: ancestral (DDPM), DDIM, and classifier-free guidance
# ---------------------------------------------------------------------------


def predict_eps_cfg(model: EpsMLP, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor | None,
                    guidance_scale: float) -> torch.Tensor:
    """Classifier-free guidance ``ε̃ = (1 + w) ε_θ(x_t, y) − w ε_θ(x_t, ∅)``  (w = 0: plain conditional).

    Args:
        x_t: (B, d).  t: (B,).  y: (B,) or None.
    Returns:
        ε̃: (B, d).
    """
    if y is None or guidance_scale == 0.0:
        return model(x_t, t, y)                                                   # (B, d)
    eps_cond = model(x_t, t, y)                                                   # (B, d)
    eps_uncond = model(x_t, t, None)                                              # (B, d)
    return (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond        # (B, d)


@torch.no_grad()
def sample_ancestral(model: EpsMLP, sched: NoiseSchedule, n: int, d: int, y: torch.Tensor | None = None,
                     guidance_scale: float = 0.0) -> torch.Tensor:
    """DDPM sampling: for t = T−1 … 0,  x_{t−1} = μ̃_t(x_t, x̂_0) + sqrt(β̃_t) z.

    Returns:
        x_0: (N, d).
    """
    x = torch.randn(n, d)                                                         # (N, d)  x_T ~ N(0, I)
    for step in reversed(range(sched.T)):
        t = torch.full((n,), step, dtype=torch.long)                              # (N,)
        eps_hat = predict_eps_cfg(model, x, t, y, guidance_scale)                 # (N, d)
        x0_hat = eps_to_x0(sched, x, t, eps_hat)                                  # (N, d)
        mean, var = posterior_mean_variance(sched, x, x0_hat, t)                  # (N, d), (N, 1)
        noise = torch.randn_like(x) if step > 0 else torch.zeros_like(x)          # (N, d)  no noise at t = 0
        x = mean + var.sqrt() * noise                                             # (N, d)
    return x


@torch.no_grad()
def sample_ddim(model: EpsMLP, sched: NoiseSchedule, n: int, d: int, n_steps: int, eta: float = 0.0,
                y: torch.Tensor | None = None, guidance_scale: float = 0.0) -> torch.Tensor:
    """DDIM sampling on a sub-sequence of ``n_steps`` timesteps (η = 0 is deterministic).

    x_{τ_{i−1}} = sqrt(ᾱ_{τ_{i−1}}) x̂_0 + sqrt(1 − ᾱ_{τ_{i−1}} − σ²) ε̂ + σ z,
    σ = η sqrt((1 − ᾱ_prev)/(1 − ᾱ)) sqrt(1 − ᾱ/ᾱ_prev).

    Returns:
        x_0: (N, d).
    """
    taus = torch.linspace(0, sched.T - 1, n_steps).round().long()                 # (S,) increasing
    x = torch.randn(n, d)                                                         # (N, d)
    for i in reversed(range(n_steps)):
        t = taus[i].repeat(n)                                                     # (N,)
        ab = sched.alpha_bar[taus[i]]                                             # scalar
        ab_prev = sched.alpha_bar[taus[i - 1]] if i > 0 else torch.tensor(1.0)    # scalar
        eps_hat = predict_eps_cfg(model, x, t, y, guidance_scale)                 # (N, d)
        x0_hat = (x - (1 - ab).sqrt() * eps_hat) / ab.sqrt()                      # (N, d)
        sigma = eta * ((1 - ab_prev) / (1 - ab)).sqrt() * (1 - ab / ab_prev).sqrt()  # scalar
        dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps_hat       # (N, d)  "direction pointing to x_t"
        noise = sigma * torch.randn_like(x) if i > 0 else torch.zeros_like(x)     # (N, d)
        x = ab_prev.sqrt() * x0_hat + dir_xt + noise                              # (N, d)
    return x


def score_from_eps(sched: NoiseSchedule, eps_hat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``∇_x log p_t(x_t) ≈ −ε_θ / sqrt(1 − ᾱ_t)`` (Tweedie / denoising score matching).  Shapes (B, d)."""
    s = gather(sched.sqrt_one_minus_alpha_bar, t, eps_hat.ndim)                   # (B, 1)
    return -eps_hat / s
