# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/optim/clipping.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k clipping -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py optim/clipping --force

"""Gradient clipping (global norm and per-element value), NumPy, in place."""
from __future__ import annotations
import numpy as np

def global_grad_norm(grads: list[np.ndarray]) -> float:
    """``||g||_2`` over the concatenation of *all* gradients: ``sqrt(sum_i ||g_i||^2)``."""
    raise NotImplementedError('TODO: implement global_grad_norm (see the reference in src/mlbook)')

def clip_grad_norm(grads: list[np.ndarray], max_norm: float) -> float:
    """Scale every gradient by ``min(1, max_norm / ||g||)`` so the *global* norm is at most ``max_norm``.

    Matches ``torch.nn.utils.clip_grad_norm_``: the direction is preserved, only
    the magnitude is capped. Modifies ``grads`` in place.

    Args:
        grads: list of arrays of any shapes.
    Returns:
        the pre-clip global norm (what you log to spot spikes).
    """
    raise NotImplementedError('TODO: implement clip_grad_norm (see the reference in src/mlbook)')

def clip_grad_value(grads: list[np.ndarray], max_value: float) -> None:
    """Elementwise ``g <- clip(g, -max_value, max_value)``. Changes the direction; rarely used for LLMs."""
    raise NotImplementedError('TODO: implement clip_grad_value (see the reference in src/mlbook)')
