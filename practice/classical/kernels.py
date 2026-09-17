# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/kernels.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k kernels -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/kernels --force

"""Kernels, kernel ridge regression and Nadaraya–Watson kernel smoothing.

A kernel ``κ(x, x')`` is an inner product in a feature space:
``κ(x, x') = φ(x)·φ(x')``. Mercer's condition (the Gram matrix ``K_ij = κ(x_i, x_j)``
is PSD for every finite set) is what makes that feature space exist.
"""
from __future__ import annotations
import numpy as np

def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``K_ij = a_i · b_j``. ``A``: (N, d), ``B``: (M, d) -> (N, M)."""
    raise NotImplementedError('TODO: implement linear_kernel (see the reference in src/mlbook)')

def polynomial_kernel(A: np.ndarray, B: np.ndarray, degree: int=3, coef0: float=1.0, gamma: float=1.0) -> np.ndarray:
    """``K_ij = (γ a_i·b_j + c)^p``. Feature space = all monomials up to degree ``p``. (N, M)."""
    raise NotImplementedError('TODO: implement polynomial_kernel (see the reference in src/mlbook)')

def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float=1.0) -> np.ndarray:
    """``K_ij = exp(-γ ||a_i - b_j||²)``: infinite-dimensional feature space. (N, M)."""
    raise NotImplementedError('TODO: implement rbf_kernel (see the reference in src/mlbook)')

class KernelRidge:
    """Kernel ridge regression: ``α = (K + λ I)^{-1} y``, ``f(x) = Σ_i α_i κ(x_i, x)``.

    Derivation: minimise ``||Φw - y||² + λ||w||²`` with ``w = Φ^T α`` (representer
    theorem) and ``K = Φ Φ^T``. Training is ``O(N³)``; prediction ``O(N)`` per query.
    ``fit(X (N, d), y (N,))``; ``predict(Q (M, d)) -> (M,)``.
    """

    def __init__(self, kernel, lam: float=1.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelRidge':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

def nadaraya_watson(Q: np.ndarray, X: np.ndarray, y: np.ndarray, kernel) -> np.ndarray:
    """Kernel smoother ``f(q) = Σ_i κ(q, x_i) y_i / Σ_i κ(q, x_i)``.

    This is exactly attention with ``q`` as query, ``x_i`` as keys, ``y_i`` as values
    and ``κ`` as the (unnormalised) score; softmax attention uses ``κ = exp(q·k/√d)``.
    ``Q``: (M, d), ``X``: (N, d), ``y``: (N, v) or (N,) -> (M, v) or (M,).
    """
    raise NotImplementedError('TODO: implement nadaraya_watson (see the reference in src/mlbook)')
