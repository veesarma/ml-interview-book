# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/decision_tree.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k decision_tree -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/decision_tree --force

"""CART decision trees from scratch (classification and regression).

Impurity criteria: entropy, Gini (classification) and variance (regression).
Splits are axis-aligned thresholds chosen greedily; each split costs
``O(d · N log N)`` with the sort-and-scan trick used in :func:`best_split`.

Conventions: ``X`` (N, d), ``y`` (N,) integer labels in ``{0..K-1}`` for
classification or floats for regression.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def class_proportions(y: np.ndarray, n_classes: int) -> np.ndarray:
    """``y``: (n,) labels -> (K,) proportions ``p_k = n_k / n``."""
    raise NotImplementedError('TODO: implement class_proportions (see the reference in src/mlbook)')

def entropy(p: np.ndarray) -> float:
    """``H(p) = -Σ_k p_k log2 p_k`` with ``0 log 0 = 0``. ``p``: (K,)."""
    raise NotImplementedError('TODO: implement entropy (see the reference in src/mlbook)')

def gini(p: np.ndarray) -> float:
    """``G(p) = 1 - Σ_k p_k^2`` = probability two random draws disagree. ``p``: (K,)."""
    raise NotImplementedError('TODO: implement gini (see the reference in src/mlbook)')

def variance(y: np.ndarray) -> float:
    """Mean squared deviation from the mean. ``y``: (n,) floats."""
    raise NotImplementedError('TODO: implement variance (see the reference in src/mlbook)')

def _node_impurity(y: np.ndarray, task: str, criterion: str, n_classes: int) -> float:
    raise NotImplementedError('TODO: implement _node_impurity (see the reference in src/mlbook)')

def best_split(X: np.ndarray, y: np.ndarray, task: str, criterion: str, n_classes: int, feature_indices: np.ndarray | None=None, min_samples_leaf: int=1) -> tuple[int, float, float] | None:
    """Return ``(feature, threshold, gain)`` of the best axis-aligned split, or ``None``.

    Gain = ``I(parent) - (n_L/n) I(left) - (n_R/n) I(right)``: information gain
    for entropy, Gini decrease for Gini, variance reduction for regression.
    Candidate thresholds are midpoints between consecutive distinct sorted values.

    ``X``: (n, d), ``y``: (n,). ``feature_indices``: optional subset of columns
    (used by random forests).
    """
    raise NotImplementedError('TODO: implement best_split (see the reference in src/mlbook)')

@dataclass
class Node:
    """A tree node. Leaves carry ``value``: class proportions (K,) or the mean (float)."""
    feature: int = -1
    threshold: float = 0.0
    left: 'Node | None' = None
    right: 'Node | None' = None
    value: np.ndarray | float | None = None

    @property
    def is_leaf(self) -> bool:
        raise NotImplementedError('TODO: implement is_leaf (see the reference in src/mlbook)')

class DecisionTree:
    """CART tree. ``task`` in {"classification", "regression"};
    ``criterion`` in {"gini", "entropy"} (ignored for regression).

    ``fit(X (N, d), y (N,))``; ``predict(X (N, d)) -> (N,)``;
    ``predict_proba(X) -> (N, K)`` for classification.
    """

    def __init__(self, task: str='classification', criterion: str='gini', max_depth: int=5, min_samples_leaf: int=1, min_gain: float=0.0, max_features: int | None=None, rng: np.random.Generator | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _leaf(self, y: np.ndarray) -> Node:
        raise NotImplementedError('TODO: implement _leaf (see the reference in src/mlbook)')

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        raise NotImplementedError('TODO: implement _grow (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def _predict_one(self, x: np.ndarray) -> np.ndarray | float:
        raise NotImplementedError('TODO: implement _predict_one (see the reference in src/mlbook)')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``X``: (N, d) -> (N, K) leaf class proportions."""
        raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``X``: (N, d) -> (N,) labels (argmax of leaf proportions) or leaf means."""
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

    def depth(self) -> int:
        raise NotImplementedError('TODO: implement depth (see the reference in src/mlbook)')
