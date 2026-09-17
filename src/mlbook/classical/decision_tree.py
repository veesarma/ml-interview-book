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


# --------------------------------------------------------------------------- #
# Impurities
# --------------------------------------------------------------------------- #
def class_proportions(y: np.ndarray, n_classes: int) -> np.ndarray:
    """``y``: (n,) labels -> (K,) proportions ``p_k = n_k / n``."""
    counts = np.bincount(y, minlength=n_classes).astype(float)  # (K,)
    return counts / max(len(y), 1)  # (K,)


def entropy(p: np.ndarray) -> float:
    """``H(p) = -Σ_k p_k log2 p_k`` with ``0 log 0 = 0``. ``p``: (K,)."""
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def gini(p: np.ndarray) -> float:
    """``G(p) = 1 - Σ_k p_k^2`` = probability two random draws disagree. ``p``: (K,)."""
    return float(1.0 - (p * p).sum())


def variance(y: np.ndarray) -> float:
    """Mean squared deviation from the mean. ``y``: (n,) floats."""
    if len(y) == 0:
        return 0.0
    return float(((y - y.mean()) ** 2).mean())


# --------------------------------------------------------------------------- #
# Split search
# --------------------------------------------------------------------------- #
def _node_impurity(y: np.ndarray, task: str, criterion: str, n_classes: int) -> float:
    if task == "regression":
        return variance(y)
    p = class_proportions(y, n_classes)  # (K,)
    return entropy(p) if criterion == "entropy" else gini(p)


def best_split(
    X: np.ndarray, y: np.ndarray, task: str, criterion: str, n_classes: int,
    feature_indices: np.ndarray | None = None, min_samples_leaf: int = 1,
) -> tuple[int, float, float] | None:
    """Return ``(feature, threshold, gain)`` of the best axis-aligned split, or ``None``.

    Gain = ``I(parent) - (n_L/n) I(left) - (n_R/n) I(right)``: information gain
    for entropy, Gini decrease for Gini, variance reduction for regression.
    Candidate thresholds are midpoints between consecutive distinct sorted values.

    ``X``: (n, d), ``y``: (n,). ``feature_indices``: optional subset of columns
    (used by random forests).
    """
    n, d = X.shape
    parent = _node_impurity(y, task, criterion, n_classes)
    features = np.arange(d) if feature_indices is None else feature_indices  # (d',)
    best: tuple[int, float, float] | None = None
    for j in features:
        order = np.argsort(X[:, j], kind="stable")  # (n,)
        xs = X[order, j]  # (n,) sorted feature values
        ys = y[order]  # (n,) labels in that order
        for i in range(min_samples_leaf, n - min_samples_leaf + 1):
            if i == 0 or i == n or xs[i - 1] == xs[i]:
                continue  # cannot split between equal values
            left, right = ys[:i], ys[i:]  # (i,), (n-i,)
            child = (i / n) * _node_impurity(left, task, criterion, n_classes) + (
                (n - i) / n
            ) * _node_impurity(right, task, criterion, n_classes)
            gain = parent - child
            if best is None or gain > best[2]:
                best = (int(j), float(0.5 * (xs[i - 1] + xs[i])), float(gain))
    return best


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """A tree node. Leaves carry ``value``: class proportions (K,) or the mean (float)."""

    feature: int = -1
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None
    value: np.ndarray | float | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class DecisionTree:
    """CART tree. ``task`` in {"classification", "regression"};
    ``criterion`` in {"gini", "entropy"} (ignored for regression).

    ``fit(X (N, d), y (N,))``; ``predict(X (N, d)) -> (N,)``;
    ``predict_proba(X) -> (N, K)`` for classification.
    """

    def __init__(
        self, task: str = "classification", criterion: str = "gini", max_depth: int = 5,
        min_samples_leaf: int = 1, min_gain: float = 0.0, max_features: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_features = max_features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.root: Node | None = None
        self.n_classes = 0

    def _leaf(self, y: np.ndarray) -> Node:
        if self.task == "regression":
            return Node(value=float(y.mean()))
        return Node(value=class_proportions(y, self.n_classes))  # value: (K,)

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n, d = X.shape
        pure = (self.task == "classification" and len(np.unique(y)) == 1) or (
            self.task == "regression" and variance(y) == 0.0
        )
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf or pure:
            return self._leaf(y)
        feats = None
        if self.max_features is not None and self.max_features < d:
            feats = self.rng.choice(d, size=self.max_features, replace=False)  # (max_features,)
        split = best_split(X, y, self.task, self.criterion, self.n_classes, feats, self.min_samples_leaf)
        if split is None or split[2] <= self.min_gain:
            return self._leaf(y)
        j, t, _ = split
        mask = X[:, j] <= t  # (n,) bool
        node = Node(feature=j, threshold=t)
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        if self.task == "classification":
            y = y.astype(int)
            self.n_classes = int(y.max()) + 1
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray) -> np.ndarray | float:
        node = self.root
        while not node.is_leaf:  # type: ignore[union-attr]
            node = node.left if x[node.feature] <= node.threshold else node.right  # type: ignore[union-attr]
        return node.value  # type: ignore[union-attr,return-value]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``X``: (N, d) -> (N, K) leaf class proportions."""
        return np.stack([self._predict_one(x) for x in X], axis=0)  # (N, K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``X``: (N, d) -> (N,) labels (argmax of leaf proportions) or leaf means."""
        if self.task == "regression":
            return np.array([self._predict_one(x) for x in X])  # (N,)
        return self.predict_proba(X).argmax(axis=1)  # (N,)

    def depth(self) -> int:
        def _d(node: Node | None) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))

        return _d(self.root)
