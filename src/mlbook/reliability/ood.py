"""Out-of-distribution scores (NumPy): max-softmax, energy, Mahalanobis on features.

Convention: every score is oriented so that LARGER means MORE in-distribution.
    max-softmax   max_k softmax(z)_k
    energy        -E(z) = T * logsumexp(z / T)             (Liu et al. 2020)
    Mahalanobis   -min_k (f - mu_k)^T Sigma^{-1} (f - mu_k)   (Lee et al. 2018)
"""
from __future__ import annotations

import numpy as np


def _logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def max_softmax_score(logits: np.ndarray) -> np.ndarray:
    """logits (N, K) -> max softmax probability (N,)."""
    z = logits - logits.max(axis=1, keepdims=True)  # (N, K)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)  # (N, K)
    return p.max(axis=1)  # (N,)


def energy_score(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    """Negative free energy T * logsumexp(z / T). logits (N, K) -> (N,)."""
    return T * _logsumexp(logits / T, axis=1)  # (N,)


class MahalanobisDetector:
    """Class-conditional Gaussians with a shared covariance on penultimate features."""

    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self.means: np.ndarray | None = None  # (K, d)
        self.prec: np.ndarray | None = None  # (d, d)

    def fit(self, F: np.ndarray, y: np.ndarray) -> "MahalanobisDetector":
        """F (N, d) features, y (N,) labels in [0, K)."""
        K = int(y.max()) + 1
        d = F.shape[1]
        self.means = np.stack([F[y == k].mean(axis=0) for k in range(K)])  # (K, d)
        centred = F - self.means[y]  # (N, d)
        cov = centred.T @ centred / len(F) + self.reg * np.eye(d)  # (d, d)
        self.prec = np.linalg.inv(cov)  # (d, d)
        return self

    def score(self, F: np.ndarray) -> np.ndarray:
        """-min_k Mahalanobis^2. F (N, d) -> (N,)."""
        assert self.means is not None and self.prec is not None
        diff = F[:, None, :] - self.means[None, :, :]  # (N, K, d)
        m2 = np.einsum("nkd,de,nke->nk", diff, self.prec, diff)  # (N, K): diff^T Sigma^-1 diff per class
        return -m2.min(axis=1)  # (N,)
