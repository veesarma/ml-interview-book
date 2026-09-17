"""Random forest: bagging + per-split feature subsampling + out-of-bag error.

Why the variance drops: averaging ``M`` estimators with variance ``σ²`` and
pairwise correlation ``ρ`` gives ``Var = ρ σ² + (1 - ρ) σ² / M``. Bagging alone
leaves ``ρ`` high (the same strong features dominate every tree); subsampling
features at each split de-correlates the trees and drives the first term down.
"""

from __future__ import annotations

import numpy as np

from .decision_tree import DecisionTree


class RandomForest:
    """Bagged CART trees with feature subsampling.

    ``fit(X (N, d), y (N,))``; ``predict(X (N, d)) -> (N,)``.
    After ``fit``, ``oob_score_`` is the out-of-bag accuracy (classification) or
    out-of-bag R² (regression).
    """

    def __init__(
        self, n_trees: int = 50, task: str = "classification", max_depth: int = 8,
        max_features: int | str | None = "sqrt", min_samples_leaf: int = 1, seed: int = 0,
    ) -> None:
        self.n_trees = n_trees
        self.task = task
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.default_rng(seed)
        self.trees: list[DecisionTree] = []
        self.oob_score_: float | None = None
        self.n_classes = 0

    def _n_features(self, d: int) -> int:
        if self.max_features is None:
            return d
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        return int(self.max_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        N, d = X.shape
        if self.task == "classification":
            y = y.astype(int)
            self.n_classes = int(y.max()) + 1
            oob_votes = np.zeros((N, self.n_classes))  # (N, K) accumulated OOB proportions
        else:
            oob_votes = np.zeros(N)  # (N,) accumulated OOB predictions
        oob_counts = np.zeros(N)  # (N,) how many trees saw each example as OOB
        self.trees = []
        for m in range(self.n_trees):
            idx = self.rng.integers(0, N, size=N)  # (N,) bootstrap sample with replacement
            oob = np.setdiff1d(np.arange(N), idx)  # (N_oob,)
            tree = DecisionTree(
                task=self.task, max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                max_features=self._n_features(d), rng=np.random.default_rng(self.rng.integers(1 << 31)),
            )
            tree.n_classes = self.n_classes
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            if len(oob) > 0:
                if self.task == "classification":
                    oob_votes[oob] += tree.predict_proba(X[oob])  # (N_oob, K)
                else:
                    oob_votes[oob] += tree.predict(X[oob])  # (N_oob,)
                oob_counts[oob] += 1
        seen = oob_counts > 0  # (N,) bool
        if self.task == "classification":
            pred = oob_votes[seen].argmax(axis=1)  # (N_seen,)
            self.oob_score_ = float((pred == y[seen]).mean())
        else:
            pred = oob_votes[seen] / oob_counts[seen]  # (N_seen,)
            ss_res = ((y[seen] - pred) ** 2).sum()
            ss_tot = ((y[seen] - y[seen].mean()) ** 2).sum()
            self.oob_score_ = float(1.0 - ss_res / ss_tot)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average of per-tree leaf proportions. ``X``: (N, d) -> (N, K)."""
        probs = np.zeros((X.shape[0], self.n_classes))  # (N, K)
        for tree in self.trees:
            probs += tree.predict_proba(X)  # (N, K)
        return probs / len(self.trees)  # (N, K)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``X``: (N, d) -> (N,): majority vote (classification) or mean (regression)."""
        if self.task == "classification":
            return self.predict_proba(X).argmax(axis=1)  # (N,)
        preds = np.stack([tree.predict(X) for tree in self.trees], axis=0)  # (M, N)
        return preds.mean(axis=0)  # (N,)
