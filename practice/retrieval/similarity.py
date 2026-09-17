# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/similarity.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k similarity -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/similarity --force

"""Similarity functions and the MIPS -> nearest-neighbour reduction (NumPy).

Conventions: row-major data, one vector per row.
    Q: (Nq, d) queries, X: (N, d) corpus.

Equations implemented
    dot(q, x)    = q . x
    cos(q, x)    = q . x / (||q|| ||x||)
    ||q - x||^2  = ||q||^2 + ||x||^2 - 2 q . x
When rows are L2-normalised all three orderings coincide:
    ||q - x||^2 = 2 - 2 cos(q, x).
"""
from __future__ import annotations
import numpy as np

def normalize_rows(X: np.ndarray, eps: float=1e-12) -> np.ndarray:
    """L2-normalise each row. (N, d) -> (N, d)."""
    raise NotImplementedError('TODO: implement normalize_rows (see the reference in src/mlbook)')

def dot_product(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Inner products. Q (Nq, d), X (N, d) -> (Nq, N)."""
    raise NotImplementedError('TODO: implement dot_product (see the reference in src/mlbook)')

def cosine_similarity(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cosine similarity. Q (Nq, d), X (N, d) -> (Nq, N)."""
    raise NotImplementedError('TODO: implement cosine_similarity (see the reference in src/mlbook)')

def squared_euclidean(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Squared L2 distance via the expansion ||q||^2 + ||x||^2 - 2 q.x.

    Q (Nq, d), X (N, d) -> (Nq, N). Clipped at 0 to absorb rounding error.
    """
    raise NotImplementedError('TODO: implement squared_euclidean (see the reference in src/mlbook)')

def mips_to_nn_corpus(X: np.ndarray) -> np.ndarray:
    """Augment the corpus so that inner-product search becomes L2 search.

    Shrivastava & Li (2014) / Bachrach et al. (2014): with M = max_i ||x_i||,
        x' = [x, sqrt(M^2 - ||x||^2)]      (all x' have norm M)
        q' = [q, 0]
        ||q' - x'||^2 = ||q||^2 + M^2 - 2 q.x
    so argmin_x ||q' - x'|| = argmax_x q.x.   X (N, d) -> (N, d + 1).
    """
    raise NotImplementedError('TODO: implement mips_to_nn_corpus (see the reference in src/mlbook)')

def mips_to_nn_query(Q: np.ndarray) -> np.ndarray:
    """Append a zero coordinate to each query. Q (Nq, d) -> (Nq, d + 1)."""
    raise NotImplementedError('TODO: implement mips_to_nn_query (see the reference in src/mlbook)')

def brute_force_topk(Q: np.ndarray, X: np.ndarray, k: int, metric: str='dot') -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by scanning the whole corpus.

    metric in {"dot", "cosine", "l2"}. Returns (ids (Nq, k), scores (Nq, k)) ordered
    best-first; for "l2" the score is the squared distance (smaller is better).
    """
    raise NotImplementedError('TODO: implement brute_force_topk (see the reference in src/mlbook)')
