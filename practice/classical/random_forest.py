# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/random_forest.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k random_forest -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/random_forest --force

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

    def __init__(self, n_trees: int=50, task: str='classification', max_depth: int=8, max_features: int | str | None='sqrt', min_samples_leaf: int=1, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _n_features(self, d: int) -> int:
        raise NotImplementedError('TODO: implement _n_features (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average of per-tree leaf proportions. ``X``: (N, d) -> (N, K)."""
        raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """``X``: (N, d) -> (N,): majority vote (classification) or mean (regression)."""
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')
