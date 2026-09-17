# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/gmm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gmm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/gmm --force

"""Gaussian mixture model fitted with EM.

E-step: responsibilities ``r_ik = π_k N(x_i; μ_k, Σ_k) / Σ_j π_j N(x_i; μ_j, Σ_j)``.
M-step: ``N_k = Σ_i r_ik``, ``π_k = N_k / N``, ``μ_k = (1/N_k) Σ_i r_ik x_i``,
``Σ_k = (1/N_k) Σ_i r_ik (x_i - μ_k)(x_i - μ_k)^T``.
The log-likelihood is non-decreasing across iterations (tracked in ``history_``).
"""
from __future__ import annotations
import numpy as np
from .kmeans import kmeans_plusplus_init

def log_gaussian(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """``log N(x_i; μ, Σ)`` for every row. ``X``: (N, d), ``mu``: (d,), ``Sigma``: (d, d) -> (N,)."""
    raise NotImplementedError('TODO: implement log_gaussian (see the reference in src/mlbook)')

def logsumexp(A: np.ndarray, axis: int=1) -> np.ndarray:
    """Stable ``log Σ exp`` along ``axis``."""
    raise NotImplementedError('TODO: implement logsumexp (see the reference in src/mlbook)')

class GMM:
    """``fit(X (N, d))``; ``predict_proba(X) -> (N, k)`` responsibilities;
    ``score(X) -> float`` mean log-likelihood; ``predict(X) -> (N,)`` hard labels."""

    def __init__(self, k: int, n_iters: int=100, reg: float=1e-06, seed: int=0, tol: float=1e-08) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        """``log π_k + log N(x_i; μ_k, Σ_k)``. ``X``: (N, d) -> (N, k)."""
        raise NotImplementedError('TODO: implement _log_joint (see the reference in src/mlbook)')

    def e_step(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Responsibilities (N, k) and the total log-likelihood ``Σ_i log Σ_k π_k N(...)``."""
        raise NotImplementedError('TODO: implement e_step (see the reference in src/mlbook)')

    def m_step(self, X: np.ndarray, R: np.ndarray) -> None:
        """Closed-form updates given responsibilities ``R`` (N, k)."""
        raise NotImplementedError('TODO: implement m_step (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray) -> 'GMM':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

    def score(self, X: np.ndarray) -> float:
        """Mean per-example log-likelihood."""
        raise NotImplementedError('TODO: implement score (see the reference in src/mlbook)')
