# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/svm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k svm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/svm --force

"""Support vector machines: the soft-margin dual solved with a simplified SMO,
and the primal hinge-loss formulation solved by (sub)gradient descent.

Dual (labels ``y ∈ {-1, +1}``):

    max_α  Σ_i α_i - ½ Σ_ij α_i α_j y_i y_j κ(x_i, x_j)
    s.t.   0 ≤ α_i ≤ C,   Σ_i α_i y_i = 0.

``f(x) = Σ_i α_i y_i κ(x_i, x) + b``; support vectors are the ``i`` with ``α_i > 0``.
"""
from __future__ import annotations
import numpy as np

class SVM:
    """Kernel soft-margin SVM trained with SMO-lite (Platt 1998, random second index).

    ``fit(X (N, d), y (N,) in {-1,+1})``; ``decision_function(Q (M, d)) -> (M,)``;
    ``predict(Q) -> (M,)`` in {-1,+1}.
    """

    def __init__(self, kernel, C: float=1.0, max_passes: int=10, tol: float=0.001, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _f(self, K: np.ndarray, i: int) -> float:
        """``f(x_i) = Σ_j α_j y_j K_ji + b``."""
        raise NotImplementedError('TODO: implement _f (see the reference in src/mlbook)')

    def _update_pair(self, K: np.ndarray, i: int, j: int) -> bool:
        """Optimise ``(α_i, α_j)`` analytically with all other α fixed. Returns True if changed."""
        raise NotImplementedError('TODO: implement _update_pair (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    @property
    def support_(self) -> np.ndarray:
        """Indices with ``α_i > 0``."""
        raise NotImplementedError('TODO: implement support_ (see the reference in src/mlbook)')

    def decision_function(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement decision_function (see the reference in src/mlbook)')

    def predict(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

def dual_objective(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """``Σ α_i - ½ Σ_ij α_i α_j y_i y_j K_ij``. ``alpha``, ``y``: (N,), ``K``: (N, N)."""
    raise NotImplementedError('TODO: implement dual_objective (see the reference in src/mlbook)')

def hinge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, C: float) -> float:
    """Primal soft-margin objective ``½||w||² + C Σ_i max(0, 1 - y_i (w·x_i + b))``."""
    raise NotImplementedError('TODO: implement hinge_loss (see the reference in src/mlbook)')

def fit_linear_svm_primal(X: np.ndarray, y: np.ndarray, C: float=1.0, lr: float=0.001, n_steps: int=5000) -> tuple[np.ndarray, float]:
    """Subgradient descent on the primal hinge objective (the route that scales to huge N).

    Subgradient: ``w - C Σ_{i: margin_i < 1} y_i x_i``; ``-C Σ_{i: margin_i < 1} y_i`` for ``b``.
    ``X``: (N, d), ``y``: (N,) in {-1,+1} -> ``(w (d,), b)``.
    """
    raise NotImplementedError('TODO: implement fit_linear_svm_primal (see the reference in src/mlbook)')
