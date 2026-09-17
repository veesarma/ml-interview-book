"""Active-learning acquisition: uncertainty (least confidence, margin, entropy), diversity
(k-center greedy / core-set) and BADGE (k-means++ on gradient embeddings).
"""

from __future__ import annotations

import numpy as np


def least_confidence(probs: np.ndarray) -> np.ndarray:
    """``1 − max_k p_k``.  probs: (N, K) → (N,); larger = more uncertain."""
    return 1.0 - probs.max(axis=1)


def margin_uncertainty(probs: np.ndarray) -> np.ndarray:
    """``−(p_(1) − p_(2))`` (negative top-two margin).  probs: (N, K) → (N,)."""
    top2 = np.sort(probs, axis=1)[:, -2:]                         # (N, 2)
    return -(top2[:, 1] - top2[:, 0])


def entropy_uncertainty(probs: np.ndarray) -> np.ndarray:
    """``H(p) = −Σ_k p_k log p_k``.  probs: (N, K) → (N,)."""
    return -(probs * np.log(probs + 1e-12)).sum(axis=1)


def k_center_greedy(features: np.ndarray, labeled_idx: np.ndarray, budget: int) -> np.ndarray:
    """Core-set selection: repeatedly pick the point farthest from the current labelled set.

    Args:
        features: (N, d).  labeled_idx: (n_l,) indices already labelled.
    Returns:
        (budget,) indices of the new points to label.
    """
    N = features.shape[0]
    if labeled_idx.size == 0:
        min_dist = np.full(N, np.inf)                             # (N,)
    else:
        diff = features[:, None, :] - features[labeled_idx][None, :, :]  # (N, n_l, d)
        min_dist = np.linalg.norm(diff, axis=2).min(axis=1)       # (N,) distance to nearest labelled
    chosen: list[int] = []
    for _ in range(budget):
        i = int(np.argmax(min_dist))
        chosen.append(i)
        d_new = np.linalg.norm(features - features[i][None, :], axis=1)  # (N,)
        min_dist = np.minimum(min_dist, d_new)                    # (N,)
    return np.array(chosen)


def badge_gradient_embeddings(probs: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Gradient of the CE loss w.r.t. the last linear layer, using the argmax as pseudo-label.

    For last layer logits = W h, the gradient w.r.t. W is (p − e_ŷ) ⊗ h, flattened to (N, K·d).
    Its norm is large when the model is uncertain (p far from one-hot) — uncertainty for free —
    and its direction depends on h — diversity for free.
    """
    N, K = probs.shape
    y_hat = probs.argmax(axis=1)                                  # (N,)
    one_hot = np.eye(K)[y_hat]                                    # (N, K)
    delta = probs - one_hot                                       # (N, K)
    g = delta[:, :, None] * features[:, None, :]                  # (N, K, d)
    return g.reshape(N, -1)                                       # (N, K·d)


def kmeans_plus_plus_seeding(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialisation: sample points with probability ∝ squared distance to chosen set.

    Args:
        x: (N, d).
    Returns:
        (k,) chosen indices.
    """
    N = x.shape[0]
    first = int(rng.integers(N))
    chosen = [first]
    d2 = ((x - x[first][None, :]) ** 2).sum(axis=1)               # (N,)
    for _ in range(k - 1):
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(N, 1.0 / N)  # (N,)
        i = int(rng.choice(N, p=p))
        chosen.append(i)
        d2 = np.minimum(d2, ((x - x[i][None, :]) ** 2).sum(axis=1))  # (N,)
    return np.array(chosen)


def badge_select(probs: np.ndarray, features: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """BADGE: k-means++ seeding on the gradient embeddings.  Returns (budget,) indices."""
    g = badge_gradient_embeddings(probs, features)                # (N, K·d)
    return kmeans_plus_plus_seeding(g, budget, rng)
