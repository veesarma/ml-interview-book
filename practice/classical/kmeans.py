# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/classical/kmeans.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k kmeans -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py classical/kmeans --force

"""k-means: Lloyd's algorithm, k-means++ seeding, and mini-batch k-means.

Objective: ``J(C, μ) = Σ_i ||x_i - μ_{c_i}||²``. Lloyd alternates
(1) assign each point to its nearest centroid (minimises J over ``c`` with ``μ``
fixed) and (2) move each centroid to the mean of its points (minimises J over
``μ`` with ``c`` fixed). Both steps can only lower J, and there are finitely many
assignments, so the algorithm terminates at a local minimum.
"""
from __future__ import annotations
import numpy as np

def squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``||x_i - c_j||²`` for all pairs. ``X``: (N, d), ``C``: (k, d) -> (N, k)."""
    raise NotImplementedError('TODO: implement squared_distances (see the reference in src/mlbook)')

def kmeans_objective(X: np.ndarray, C: np.ndarray, labels: np.ndarray) -> float:
    """``Σ_i ||x_i - C[labels_i]||²``. ``X``: (N, d), ``C``: (k, d), ``labels``: (N,)."""
    raise NotImplementedError('TODO: implement kmeans_objective (see the reference in src/mlbook)')

def kmeans_plusplus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur & Vassilvitskii 2007).

    First centre uniform at random; each next centre is drawn with probability
    proportional to ``D(x)²``, its squared distance to the nearest centre chosen
    so far. The expected objective is within ``O(log k)`` of optimal.

    ``X``: (N, d) -> ``C``: (k, d).
    """
    raise NotImplementedError('TODO: implement kmeans_plusplus_init (see the reference in src/mlbook)')

def kmeans(X: np.ndarray, k: int, n_iters: int=100, init: str='k-means++', seed: int=0, tol: float=1e-09) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Lloyd's algorithm. ``X``: (N, d) -> ``(C (k, d), labels (N,), objective history)``.

    The history is non-increasing (each half-step is an exact coordinate minimisation).
    """
    raise NotImplementedError('TODO: implement kmeans (see the reference in src/mlbook)')

def minibatch_kmeans(X: np.ndarray, k: int, batch_size: int=256, n_steps: int=200, seed: int=0) -> np.ndarray:
    """Mini-batch k-means (Sculley 2010): per-centre learning rate ``1 / n_j``.

    Each step draws a batch, assigns it, and moves each centre toward its
    batch members with step size ``1 / (count so far)`` — a streaming mean.

    ``X``: (N, d) -> ``C``: (k, d).
    """
    raise NotImplementedError('TODO: implement minibatch_kmeans (see the reference in src/mlbook)')
