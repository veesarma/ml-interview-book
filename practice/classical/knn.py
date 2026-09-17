# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/knn.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k knn -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/knn --force

"""k-nearest neighbours: distance metrics, brute-force search and a kd-tree.

Brute force costs ``O(N d)`` per query (plus ``O(N)`` for a partial sort);
a kd-tree is ``O(log N)`` per query in low dimension but degrades toward
brute force once ``d`` exceeds roughly 10–20 (see the chapter). Beyond that,
use approximate methods (IVF/HNSW; Part XIII).
"""
from __future__ import annotations
import numpy as np

def pairwise_distances(Q: np.ndarray, X: np.ndarray, metric: str='euclidean') -> np.ndarray:
    """Distances between every query and every stored point.

    ``Q``: (M, d), ``X``: (N, d) -> (M, N).
    euclidean: ``||q - x||`` computed as ``sqrt(||q||² + ||x||² - 2 q·x)``;
    manhattan: ``Σ |q_j - x_j|``; cosine: ``1 - q·x / (||q|| ||x||)``.
    """
    raise NotImplementedError('TODO: implement pairwise_distances (see the reference in src/mlbook)')

def knn_indices(Q: np.ndarray, X: np.ndarray, k: int, metric: str='euclidean') -> np.ndarray:
    """Indices of the ``k`` nearest stored points for each query.

    ``Q``: (M, d), ``X``: (N, d) -> (M, k), sorted by increasing distance.
    ``argpartition`` finds the k smallest in ``O(N)`` and only those are sorted.
    """
    raise NotImplementedError('TODO: implement knn_indices (see the reference in src/mlbook)')

class KNNClassifier:
    """Majority vote among the ``k`` nearest training points (optionally distance-weighted)."""

    def __init__(self, k: int=5, metric: str='euclidean', weighted: bool=False) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """Store the data: training is ``O(1)``. ``X``: (N, d), ``y``: (N,) ints."""
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict_proba(self, Q: np.ndarray) -> np.ndarray:
        """``Q``: (M, d) -> (M, K) vote fractions."""
        raise NotImplementedError('TODO: implement predict_proba (see the reference in src/mlbook)')

    def predict(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

class KNNRegressor:
    """Mean of the ``k`` nearest targets. ``fit(X (N, d), y (N,))``; ``predict(Q (M, d)) -> (M,)``."""

    def __init__(self, k: int=5, metric: str='euclidean') -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNRegressor':
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def predict(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement predict (see the reference in src/mlbook)')

class KDTree:
    """A minimal kd-tree for exact Euclidean nearest-neighbour queries.

    Build: split on the axis with the largest spread at the median, recursively.
    Query: descend to the leaf, then backtrack, pruning any subtree whose
    splitting plane is farther than the current best distance.
    """

    def __init__(self, X: np.ndarray, leaf_size: int=8) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _build(self, idx: np.ndarray) -> dict:
        raise NotImplementedError('TODO: implement _build (see the reference in src/mlbook)')

    def query(self, q: np.ndarray, k: int=1) -> tuple[np.ndarray, np.ndarray]:
        """``q``: (d,) -> (distances (k,), indices (k,)) sorted ascending."""
        raise NotImplementedError('TODO: implement query (see the reference in src/mlbook)')
