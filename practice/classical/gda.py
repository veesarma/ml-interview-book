# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/gda.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gda -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/gda --force

"""Gaussian discriminant analysis: LDA (shared covariance) and QDA (per-class).

``p(x | y=k) = N(μ_k, Σ_k)``, ``p(y=k) = π_k``. With a shared ``Σ`` the
posterior ``p(y=1 | x)`` is exactly a logistic function of a *linear* score,
which :func:`lda_as_logistic` returns explicitly.
"""
from __future__ import annotations
import numpy as np

class GDA:
    """``shared_cov=True`` gives LDA (linear boundaries); ``False`` gives QDA.

    ``fit(X (N, d), y (N,) ints)``; ``predict(X) -> (N,)``; ``log_posterior(X) -> (N, K)``.
    """

    def __init__(self, shared_cov: bool=True, reg: float=1e-06) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GDA':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def log_posterior(self, X: np.ndarray) -> np.ndarray:
        """Unnormalised ``log π_k + log N(x; μ_k, Σ_k)``. ``X``: (N, d) -> (N, K)."""
        raise NotImplementedError('TODO: implement log_posterior (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

def lda_as_logistic(model: GDA) -> tuple[np.ndarray, float]:
    """Return ``(w, b)`` with ``p(y=1 | x) = σ(w·x + b)`` for a two-class LDA.

    ``w = Σ^{-1} (μ_1 - μ_0)``,
    ``b = -½ μ_1^T Σ^{-1} μ_1 + ½ μ_0^T Σ^{-1} μ_0 + log(π_1 / π_0)``.
    """
    raise NotImplementedError('TODO: implement lda_as_logistic (see the reference in src/mlbook)')
