# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/rl/gae.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gae -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py rl/gae --force

"""Generalised Advantage Estimation (Schulman et al., 2016) in NumPy.

    delta_t = r_t + gamma (1 - d_t) V(s_{t+1}) - V(s_t)
    A_t     = sum_{l>=0} (gamma lambda)^l delta_{t+l}          (reset at episode ends)
    R_t     = A_t + V(s_t)                                     (value-function target)

``lambda = 0`` gives the one-step TD advantage (low variance, biased by V);
``lambda = 1`` gives the Monte Carlo advantage ``G_t - V(s_t)`` (unbiased, high variance).
"""
from __future__ import annotations
import numpy as np

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Backward recursion ``A_t = delta_t + gamma lambda (1 - d_t) A_{t+1}``.

    Args:
        rewards: (T,) rewards ``r_t`` for taking ``a_t`` in ``s_t``.
        values: (T,) critic values ``V(s_t)``.
        dones: (T,) 1.0 if the episode ended after step ``t`` (then ``s_{t+1}`` is terminal).
        last_value: ``V(s_T)`` used to bootstrap a truncated final segment.
    Returns:
        ``(advantages (T,), returns (T,))`` with ``returns = advantages + values``.
    """
    raise NotImplementedError('TODO: implement compute_gae (see the reference in src/mlbook)')

def n_step_advantage(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float, gamma: float, n: int) -> np.ndarray:
    """``A^(n)_t = r_t + ... + gamma^{n-1} r_{t+n-1} + gamma^n V(s_{t+n}) - V(s_t)`` for a single episode.

    Used in tests to check that GAE is the ``(1-lambda) sum_n lambda^{n-1} A^(n)`` mixture.
    Assumes ``dones`` marks at most the final step (one episode).
    """
    raise NotImplementedError('TODO: implement n_step_advantage (see the reference in src/mlbook)')
