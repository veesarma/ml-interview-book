# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/reliability/ood.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ood -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py reliability/ood --force

"""Out-of-distribution scores (NumPy): max-softmax, energy, Mahalanobis on features.

Convention: every score is oriented so that LARGER means MORE in-distribution.
    max-softmax   max_k softmax(z)_k
    energy        -E(z) = T * logsumexp(z / T)             (Liu et al. 2020)
    Mahalanobis   -min_k (f - mu_k)^T Sigma^{-1} (f - mu_k)   (Lee et al. 2018)
"""
from __future__ import annotations
import numpy as np

def _logsumexp(z: np.ndarray, axis: int=-1) -> np.ndarray:
    raise NotImplementedError('TODO: implement _logsumexp (see the reference in src/mlbook)')

def max_softmax_score(logits: np.ndarray) -> np.ndarray:
    """logits (N, K) -> max softmax probability (N,)."""
    raise NotImplementedError('TODO: implement max_softmax_score (see the reference in src/mlbook)')

def energy_score(logits: np.ndarray, T: float=1.0) -> np.ndarray:
    """Negative free energy T * logsumexp(z / T). logits (N, K) -> (N,)."""
    raise NotImplementedError('TODO: implement energy_score (see the reference in src/mlbook)')

class MahalanobisDetector:
    """Class-conditional Gaussians with a shared covariance on penultimate features."""

    def __init__(self, reg: float=0.001) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, F: np.ndarray, y: np.ndarray) -> 'MahalanobisDetector':
        """F (N, d) features, y (N,) labels in [0, K)."""
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def score(self, F: np.ndarray) -> np.ndarray:
        """-min_k Mahalanobis^2. F (N, d) -> (N,)."""
        raise NotImplementedError('TODO: implement score (see the reference in src/mlbook)')
