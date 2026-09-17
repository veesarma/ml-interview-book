"""Learning-rate schedules as pure functions of the step: ``lr = schedule(step)``.

All schedules take the integer step (0-indexed) and return a float, so they
compose with any optimizer by setting ``opt.lr = schedule(step)`` before ``step()``.
"""

from __future__ import annotations

import math


def linear_warmup(step: int, warmup_steps: int, peak_lr: float) -> float:
    """``lr = peak * min(1, (step + 1) / warmup)``: ramps linearly, then holds."""
    if warmup_steps <= 0:
        return peak_lr
    return peak_lr * min(1.0, (step + 1) / warmup_steps)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int, peak_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup then cosine decay to ``min_lr`` at ``total_steps`` (the GPT-3 / Llama schedule).

    ``lr = min + 1/2 (peak - min) (1 + cos(pi * progress))`` with
    ``progress = (step - warmup) / (total - warmup)`` clipped to [0, 1].
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps, peak_lr)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def warmup_stable_decay(
    step: int, warmup_steps: int, stable_steps: int, decay_steps: int, peak_lr: float, min_lr: float = 0.0
) -> float:
    """WSD / trapezoid schedule: warmup, constant plateau, then linear decay to ``min_lr``.

    The plateau lets you branch off a decayed checkpoint at any time without
    committing to ``total_steps`` up front (MiniCPM, DeepSeek-V3, OLMo 2 use variants).
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps, peak_lr)
    if step < warmup_steps + stable_steps:
        return peak_lr
    progress = min(1.0, (step - warmup_steps - stable_steps) / max(1, decay_steps))
    return peak_lr + (min_lr - peak_lr) * progress


def inverse_sqrt(step: int, warmup_steps: int, peak_lr: float) -> float:
    """Transformer (Vaswani et al. 2017) schedule up to a constant:
    ``lr = peak * min((step+1)/warmup, sqrt(warmup / (step+1)))``.
    """
    s = step + 1
    return peak_lr * min(s / warmup_steps, math.sqrt(warmup_steps / s))


def step_decay(step: int, base_lr: float, drop_every: int, gamma: float = 0.1) -> float:
    """Classic CV schedule: ``lr = base * gamma^(floor(step / drop_every))`` (ResNet: x0.1 at 30/60/90 epochs)."""
    return base_lr * gamma ** (step // drop_every)
