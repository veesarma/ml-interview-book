# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/losses.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k losses -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/losses --force

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
    raise NotImplementedError('TODO: implement log_softmax (see the reference in src/mlbook)')

def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis. Input ``(..., K)``, output ``(..., K)``."""
    raise NotImplementedError('TODO: implement softmax (see the reference in src/mlbook)')

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Integer labels ``(N,)`` -> one-hot ``(N, K)`` float64."""
    raise NotImplementedError('TODO: implement one_hot (see the reference in src/mlbook)')

class MSELoss:
    """``L = (1 / (N D)) sum_{n,d} (yhat - y)^2`` (mean over all elements).

    Shapes: ``yhat`` ``(N, D)``, ``y`` ``(N, D)``. ``backward`` returns
    ``dL/dyhat = 2 (yhat - y) / (N D)`` of shape ``(N, D)``.
    """

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, yhat: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class CrossEntropyLoss:
    """Softmax cross-entropy from *logits* with integer targets.

    ``L = -(1/N) sum_n log softmax(z_n)[y_n]``.
    Shapes: ``logits`` ``(N, K)``, ``y`` ``(N,)`` int. ``backward`` returns the
    boxed result of chapter 2, ``dL/dz = (p - onehot(y)) / N`` of shape ``(N, K)``.
    """

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class BCEWithLogitsLoss:
    """Binary cross-entropy from a logit: ``L = mean( softplus(z) - y z )``.

    Shapes: ``z`` ``(N,)`` or ``(N, 1)``, ``y`` same shape in {0, 1}.
    ``backward`` returns ``(sigmoid(z) - y) / N``: the same ``p - y`` pattern.
    """

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')
