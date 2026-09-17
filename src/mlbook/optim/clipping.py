"""Gradient clipping (global norm and per-element value), NumPy, in place."""

from __future__ import annotations

import numpy as np


def global_grad_norm(grads: list[np.ndarray]) -> float:
    """``||g||_2`` over the concatenation of *all* gradients: ``sqrt(sum_i ||g_i||^2)``."""
    return float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))


def clip_grad_norm(grads: list[np.ndarray], max_norm: float) -> float:
    """Scale every gradient by ``min(1, max_norm / ||g||)`` so the *global* norm is at most ``max_norm``.

    Matches ``torch.nn.utils.clip_grad_norm_``: the direction is preserved, only
    the magnitude is capped. Modifies ``grads`` in place.

    Args:
        grads: list of arrays of any shapes.
    Returns:
        the pre-clip global norm (what you log to spot spikes).
    """
    total = global_grad_norm(grads)
    scale = min(1.0, max_norm / (total + 1e-6))
    if scale < 1.0:
        for g in grads:
            g *= scale  # in place, same shape
    return total


def clip_grad_value(grads: list[np.ndarray], max_value: float) -> None:
    """Elementwise ``g <- clip(g, -max_value, max_value)``. Changes the direction; rarely used for LLMs."""
    for g in grads:
        np.clip(g, -max_value, max_value, out=g)
