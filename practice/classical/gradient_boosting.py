# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/gradient_boosting.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gradient_boosting -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/gradient_boosting --force

"""Gradient-boosted trees with Newton (second-order) leaf weights, XGBoost style.

Round ``m`` fits a tree ``f_m`` to the second-order Taylor expansion of the loss
around the current prediction ``F_{m-1}``:

    L ≈ Σ_i [ g_i f(x_i) + ½ h_i f(x_i)² ] + γ T + ½ λ Σ_leaves w_j²

with ``g_i = ∂ℓ/∂F``, ``h_i = ∂²ℓ/∂F²``. For a fixed tree structure the optimal
leaf weight and the resulting objective are

    w_j* = -G_j / (H_j + λ),      obj* = -½ Σ_j G_j² / (H_j + λ) + γ T,

so the gain of a split is ``½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ``.

Losses: ``"squared"`` (g = F - y, h = 1) and ``"logistic"`` (g = σ(F) - y,
h = σ(F)(1 - σ(F))).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .logistic_regression import sigmoid

@dataclass
class _NewtonNode:
    feature: int = -1
    threshold: float = 0.0
    left: '_NewtonNode | None' = None
    right: '_NewtonNode | None' = None
    weight: float = 0.0

    @property
    def is_leaf(self) -> bool:
        raise NotImplementedError('TODO: implement is_leaf (see the reference in src/mlbook)')

def leaf_weight(G: float, H: float, lam: float) -> float:
    """``w* = -G / (H + λ)``: the Newton step for one leaf."""
    raise NotImplementedError('TODO: implement leaf_weight (see the reference in src/mlbook)')

def split_gain(G_L: float, H_L: float, G_R: float, H_R: float, lam: float, gamma: float) -> float:
    """``½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ``."""
    raise NotImplementedError('TODO: implement split_gain (see the reference in src/mlbook)')

class NewtonTree:
    """A regression tree grown on (g, h) pairs with the XGBoost gain and leaf weights.

    ``fit(X (N, d), g (N,), h (N,))``; ``predict(X (N, d)) -> (N,)`` leaf weights.
    """

    def __init__(self, max_depth: int=3, lam: float=1.0, gamma: float=0.0, min_child_weight: float=1.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _best_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple[int, float, float] | None:
        raise NotImplementedError('TODO: implement _best_split (see the reference in src/mlbook)')

    def _grow(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> _NewtonNode:
        raise NotImplementedError('TODO: implement _grow (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> 'NewtonTree':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

class GradientBoostedTrees:
    """Newton boosting: ``F_m = F_{m-1} + η f_m`` with ``f_m`` a :class:`NewtonTree`.

    ``loss`` in {"squared", "logistic"}. ``fit(X (N, d), y (N,))``;
    ``predict_raw(X) -> (N,)`` returns ``F_M``; ``predict`` thresholds for logistic.
    ``train_loss_`` records the loss after every round (monotone for small η).
    """

    def __init__(self, n_rounds: int=100, learning_rate: float=0.1, max_depth: int=3, lam: float=1.0, gamma: float=0.0, loss: str='squared', subsample: float=1.0, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _grad_hess(self, F: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First and second derivatives of the per-example loss w.r.t. ``F``. Both (N,)."""
        raise NotImplementedError('TODO: implement _grad_hess (see the reference in src/mlbook)')

    def _loss_value(self, F: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError('TODO: implement _loss_value (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostedTrees':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """``F_M(x) = base + η Σ_m f_m(x)``. ``X``: (N, d) -> (N,)."""
        raise NotImplementedError('TODO: implement predict_raw (see the reference in src/mlbook)')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``σ(F_M(x))`` for the logistic loss. ``X``: (N, d) -> (N,)."""
        raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')
