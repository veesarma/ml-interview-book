# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/logistic_regression.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k logistic_regression -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/logistic_regression --force

"""Binary logistic regression from scratch: sigmoid, Bernoulli negative
log-likelihood, gradient ``X^T (p - y)``, Hessian ``X^T diag(p(1-p)) X``,
gradient descent, Newton / IRLS, class weighting, and Platt scaling.

Conventions: ``X`` is ``(N, d)`` (add a bias column yourself), ``y`` is ``(N,)``
with entries in ``{0, 1}``, ``w`` is ``(d,)``.
"""
from __future__ import annotations
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``σ(z) = 1 / (1 + e^{-z})``, elementwise.

    For ``z >= 0`` use ``1 / (1 + e^{-z})``; for ``z < 0`` use ``e^{z} / (1 + e^{z})``
    so the exponent is never large and positive.
    """
    raise NotImplementedError('TODO: implement sigmoid (see the reference in src/mlbook)')

def log_sigmoid(z: np.ndarray) -> np.ndarray:
    """``log σ(z) = -softplus(-z) = -log(1 + e^{-z})`` computed stably."""
    raise NotImplementedError('TODO: implement log_sigmoid (see the reference in src/mlbook)')

def bce_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, sample_weight: np.ndarray | None=None, l2: float=0.0) -> float:
    """Mean binary cross-entropy (negative Bernoulli log-likelihood) plus ``(l2/2)||w||^2``.

    ``L = -(1/N) Σ_i s_i [ y_i log σ(z_i) + (1 - y_i) log(1 - σ(z_i)) ]`` with ``z = Xw``.
    ``X``: (N, d), ``y``: (N,), ``w``: (d,), ``sample_weight``: (N,) or None.
    """
    raise NotImplementedError('TODO: implement bce_loss (see the reference in src/mlbook)')

def bce_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, sample_weight: np.ndarray | None=None, l2: float=0.0) -> np.ndarray:
    """``∇_w L = (1/N) X^T (s ⊙ (p - y)) + l2 w`` with ``p = σ(Xw)``.

    ``X``: (N, d), ``y``: (N,), ``w``: (d,) -> (d,).
    """
    raise NotImplementedError('TODO: implement bce_gradient (see the reference in src/mlbook)')

def bce_hessian(X: np.ndarray, w: np.ndarray, sample_weight: np.ndarray | None=None, l2: float=0.0) -> np.ndarray:
    """``H = (1/N) X^T diag(s ⊙ p ⊙ (1 - p)) X + l2 I``. PSD, so the loss is convex.

    ``X``: (N, d), ``w``: (d,) -> (d, d).
    """
    raise NotImplementedError('TODO: implement bce_hessian (see the reference in src/mlbook)')

def fit_logistic_gd(X: np.ndarray, y: np.ndarray, lr: float=0.1, n_steps: int=2000, l2: float=0.0, sample_weight: np.ndarray | None=None) -> np.ndarray:
    """Full-batch gradient descent on the (weighted, L2-regularised) BCE.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    raise NotImplementedError('TODO: implement fit_logistic_gd (see the reference in src/mlbook)')

def fit_logistic_newton(X: np.ndarray, y: np.ndarray, n_steps: int=20, l2: float=1e-06, sample_weight: np.ndarray | None=None, tol: float=1e-10) -> np.ndarray:
    """Newton's method (equivalently IRLS): ``w ← w - H^{-1} ∇L``.

    Each step solves a weighted least-squares problem with weights
    ``p(1-p)`` and working response ``z + (y - p) / (p(1-p))``. Quadratic
    convergence near the optimum; ``O(N d^2 + d^3)`` per step.
    A small ``l2`` keeps ``H`` invertible when the data are separable.

    ``X``: (N, d), ``y``: (N,) -> ``w``: (d,).
    """
    raise NotImplementedError('TODO: implement fit_logistic_newton (see the reference in src/mlbook)')

def balanced_class_weights(y: np.ndarray) -> np.ndarray:
    """Per-example weights ``N / (2 N_c)`` so each class contributes equally.

    ``y``: (N,) in {0,1} -> (N,).
    """
    raise NotImplementedError('TODO: implement balanced_class_weights (see the reference in src/mlbook)')

def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``σ(Xw)``. ``X``: (N, d), ``w``: (d,) -> (N,)."""
    raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

def predict_label(X: np.ndarray, w: np.ndarray, threshold: float=0.5) -> np.ndarray:
    """Threshold the probability. Choose ``threshold`` from the cost matrix, not 0.5 by reflex."""
    raise NotImplementedError('TODO: implement predict_label (see the reference in src/mlbook)')

def platt_scaling(scores: np.ndarray, y: np.ndarray, n_steps: int=50) -> tuple[float, float]:
    """Fit ``P(y=1 | s) = σ(a s + b)`` by Newton's method. Platt scaling *is*
    1-D logistic regression on the model's raw score.

    ``scores``: (N,), ``y``: (N,) in {0,1} -> ``(a, b)``.
    """
    raise NotImplementedError('TODO: implement platt_scaling (see the reference in src/mlbook)')

def one_vs_rest_fit(X: np.ndarray, y: np.ndarray, n_classes: int, **kwargs) -> np.ndarray:
    """Train ``K`` independent binary classifiers (class k vs the rest).

    ``X``: (N, d), ``y``: (N,) in {0..K-1} -> ``W``: (d, K).
    """
    raise NotImplementedError('TODO: implement one_vs_rest_fit (see the reference in src/mlbook)')

def one_vs_rest_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Argmax over the ``K`` sigmoid scores (they need not sum to 1).

    ``X``: (N, d), ``W``: (d, K) -> (N,).
    """
    raise NotImplementedError('TODO: implement one_vs_rest_predict (see the reference in src/mlbook)')
