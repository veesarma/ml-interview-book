# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/naive_bayes.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k naive_bayes -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/naive_bayes --force

"""Naive Bayes: multinomial (counts / bag-of-words) with Laplace smoothing, and
Gaussian (continuous features).

Decision rule: ``ŷ = argmax_k [ log π_k + Σ_j log p(x_j | y = k) ]`` — the
"naive" part is the conditional independence of features given the class,
which turns a ``d``-dimensional density into ``d`` one-dimensional ones.
"""
from __future__ import annotations
import numpy as np

class MultinomialNB:
    """``p(x | y=k) ∝ Π_j θ_kj^{x_j}`` with ``θ_kj = (N_kj + α) / (N_k + α d)``.

    ``fit(X (N, d) non-negative counts, y (N,) ints)``; ``predict(X) -> (N,)``.
    """

    def __init__(self, alpha: float=1.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNB':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def log_joint(self, X: np.ndarray) -> np.ndarray:
        """``log π_k + Σ_j x_j log θ_kj``. ``X``: (N, d) -> (N, K)."""
        raise NotImplementedError('TODO: implement log_joint (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

class GaussianNB:
    """Per-class, per-feature Gaussians: ``p(x_j | y=k) = N(μ_kj, σ_kj²)``.

    ``fit(X (N, d) floats, y (N,) ints)``; ``predict(X) -> (N,)``.
    """

    def __init__(self, var_smoothing: float=1e-09) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def log_joint(self, X: np.ndarray) -> np.ndarray:
        """``log π_k - ½ Σ_j [ log(2π σ_kj²) + (x_j - μ_kj)² / σ_kj² ]``. ``X``: (N, d) -> (N, K)."""
        raise NotImplementedError('TODO: implement log_joint (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')
