"""NumPy losses with explicit ``forward`` / ``backward`` (Part III, chapters 1-2).

Every loss returns a scalar that is a *mean over the batch*, so the gradient
returned by ``backward`` already carries the ``1/N`` factor.
"""
from __future__ import annotations

import numpy as np


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise ``log softmax`` over the last axis, computed stably.

    ``log p_k = z_k - m - log sum_j exp(z_j - m)`` with ``m = max_j z_j``.
    Input ``(..., K)``, output ``(..., K)``.
    """
    m = z.max(axis=-1, keepdims=True)  # (..., 1)
    shifted = z - m  # (..., K)
    lse = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))  # (..., 1)
    return shifted - lse  # (..., K)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis. Input ``(..., K)``, output ``(..., K)``."""
    return np.exp(log_softmax(z))  # (..., K)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Integer labels ``(N,)`` -> one-hot ``(N, K)`` float64."""
    out = np.zeros((y.shape[0], num_classes))  # (N, K)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


class MSELoss:
    """``L = (1 / (N D)) sum_{n,d} (yhat - y)^2`` (mean over all elements).

    Shapes: ``yhat`` ``(N, D)``, ``y`` ``(N, D)``. ``backward`` returns
    ``dL/dyhat = 2 (yhat - y) / (N D)`` of shape ``(N, D)``.
    """

    def __init__(self) -> None:
        self._diff: np.ndarray | None = None

    def forward(self, yhat: np.ndarray, y: np.ndarray) -> float:
        self._diff = yhat - y  # (N, D)
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        assert self._diff is not None
        return 2.0 * self._diff / self._diff.size  # (N, D)


class CrossEntropyLoss:
    """Softmax cross-entropy from *logits* with integer targets.

    ``L = -(1/N) sum_n log softmax(z_n)[y_n]``.
    Shapes: ``logits`` ``(N, K)``, ``y`` ``(N,)`` int. ``backward`` returns the
    boxed result of chapter 2, ``dL/dz = (p - onehot(y)) / N`` of shape ``(N, K)``.
    """

    def __init__(self) -> None:
        self._p: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        logp = log_softmax(logits)  # (N, K)
        self._p = np.exp(logp)  # (N, K)
        self._y = y  # (N,)
        n = logits.shape[0]
        return float(-logp[np.arange(n), y].mean())

    def backward(self) -> np.ndarray:
        assert self._p is not None and self._y is not None
        n, k = self._p.shape
        grad = self._p - one_hot(self._y, k)  # (N, K)  = p - y
        return grad / n  # (N, K)


class BCEWithLogitsLoss:
    """Binary cross-entropy from a logit: ``L = mean( softplus(z) - y z )``.

    Shapes: ``z`` ``(N,)`` or ``(N, 1)``, ``y`` same shape in {0, 1}.
    ``backward`` returns ``(sigmoid(z) - y) / N``: the same ``p - y`` pattern.
    """

    def __init__(self) -> None:
        self._s: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def forward(self, z: np.ndarray, y: np.ndarray) -> float:
        softplus = np.logaddexp(0.0, z)  # (N,)  = log(1 + e^z), stable
        self._s = 1.0 / (1.0 + np.exp(-z))  # (N,)
        self._y = y
        return float(np.mean(softplus - y * z))

    def backward(self) -> np.ndarray:
        assert self._s is not None and self._y is not None
        return (self._s - self._y) / self._s.size  # (N,)
