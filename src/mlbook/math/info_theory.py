"""Entropy, cross-entropy, KL, mutual information, perplexity, InfoNCE (pure NumPy).

All functions take probability vectors ``(K,)`` or batches ``(N, K)`` over the
last axis and work in nats unless ``base`` is given. ``0 log 0`` is treated as 0.
"""

from __future__ import annotations

import numpy as np


def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``x * log(y)`` with the convention ``0 * log(0) = 0`` (elementwise)."""
    out = np.zeros_like(x, dtype=np.float64)
    mask = x > 0
    out[mask] = x[mask] * np.log(y[mask])
    return out


def entropy(p: np.ndarray, base: float | None = None) -> np.ndarray:
    """``H(p) = -sum_k p_k log p_k`` over the last axis.

    Args:
        p: (K,) or (N, K) probabilities.
    Returns:
        scalar or (N,). Nats by default; ``base=2`` for bits.
    """
    h = -np.sum(_xlogy(p, p), axis=-1)
    return h / np.log(base) if base else h


def cross_entropy(p: np.ndarray, q: np.ndarray, base: float | None = None) -> np.ndarray:
    """``H(p, q) = -sum_k p_k log q_k`` (expected code length using q's code for p's data).

    Args:
        p: (..., K) true distribution. q: (..., K) model distribution.
    """
    h = -np.sum(_xlogy(p, q), axis=-1)
    return h / np.log(base) if base else h


def kl_divergence(p: np.ndarray, q: np.ndarray, base: float | None = None) -> np.ndarray:
    """``KL(p || q) = sum_k p_k log(p_k / q_k) = H(p, q) - H(p) >= 0``.

    Args:
        p, q: (..., K). Requires ``q_k > 0`` wherever ``p_k > 0``.
    """
    kl = np.sum(_xlogy(p, p) - _xlogy(p, q), axis=-1)
    return kl / np.log(base) if base else kl


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon: ``1/2 KL(p || m) + 1/2 KL(q || m)`` with ``m = (p + q)/2``. Symmetric, bounded by log 2."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def conditional_entropy(joint: np.ndarray) -> float:
    """``H(Y | X) = H(X, Y) - H(X)`` from a joint table ``joint[x, y]``.

    Args:
        joint: (Kx, Ky) probabilities summing to one.
    """
    p_x = joint.sum(axis=1)  # (Kx,) marginal of X
    return float(entropy(joint.ravel()) - entropy(p_x))


def mutual_information(joint: np.ndarray) -> float:
    """``I(X; Y) = KL(p(x, y) || p(x) p(y)) = H(Y) - H(Y | X)``.

    Args:
        joint: (Kx, Ky).
    """
    p_x = joint.sum(axis=1, keepdims=True)  # (Kx, 1)
    p_y = joint.sum(axis=0, keepdims=True)  # (1, Ky)
    independent = p_x * p_y  # (Kx, Ky) product of marginals
    return float(kl_divergence(joint.ravel(), independent.ravel()))


def gaussian_kl(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> np.ndarray:
    """``KL(N(mu1, var1) || N(mu2, var2))`` for diagonal Gaussians, summed over the last axis.

    ``= 1/2 sum [ log(var2/var1) + (var1 + (mu1 - mu2)^2)/var2 - 1 ]``  (the VAE term).
    Args:
        all (..., d).
    """
    term = np.log(var2 / var1) + (var1 + (mu1 - mu2) ** 2) / var2 - 1.0  # (..., d)
    return 0.5 * np.sum(term, axis=-1)


# ---------------------------------------------------------------------------
# Language-model quantities
# ---------------------------------------------------------------------------


def perplexity(nll_per_token_nats: np.ndarray) -> float:
    """``PPL = exp(mean token NLL)`` -- the effective branching factor.

    Args:
        nll_per_token_nats: (N_tokens,) negative log-likelihoods in nats.
    """
    return float(np.exp(np.mean(nll_per_token_nats)))


def bits_per_byte(nll_per_token_nats: np.ndarray, n_bytes: int) -> float:
    """``BPB = (sum of token NLL in bits) / n_bytes`` -- tokenizer-independent.

    Args:
        nll_per_token_nats: (N_tokens,). n_bytes: UTF-8 length of the same text.
    """
    total_bits = float(np.sum(nll_per_token_nats)) / np.log(2.0)
    return total_bits / n_bytes


def infonce_loss(z_a: np.ndarray, z_b: np.ndarray, temperature: float = 0.07) -> float:
    """Symmetric InfoNCE (the CLIP objective) on L2-normalised embeddings.

    ``L = 1/2 [CE(rows of S/tau, diag) + CE(cols of S/tau, diag)]`` with
    ``S = z_a z_b^T``. Minimising it maximises a lower bound on ``I(a; b)``:
    ``I(a; b) >= log N - L``.

    Args:
        z_a: (N, d) e.g. image embeddings. z_b: (N, d) matching text embeddings.
    """
    z_a = z_a / np.linalg.norm(z_a, axis=1, keepdims=True)  # (N, d)
    z_b = z_b / np.linalg.norm(z_b, axis=1, keepdims=True)  # (N, d)
    logits = z_a @ z_b.T / temperature  # (N, N) similarity matrix
    n = logits.shape[0]
    # log-softmax over rows (a -> b) and over columns (b -> a)
    log_p_rows = logits - _logsumexp(logits, axis=1, keepdims=True)  # (N, N)
    log_p_cols = logits - _logsumexp(logits, axis=0, keepdims=True)  # (N, N)
    diag = np.arange(n)
    loss_ab = -np.mean(log_p_rows[diag, diag])  # scalar
    loss_ba = -np.mean(log_p_cols[diag, diag])  # scalar
    return float(0.5 * (loss_ab + loss_ba))


def _logsumexp(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def entropy_bonus(logits: np.ndarray) -> float:
    """Mean policy entropy from logits ``(N, K)``; added to the PPO objective as ``+ beta H``."""
    log_p = logits - _logsumexp(logits, axis=-1, keepdims=True)  # (N, K)
    p = np.exp(log_p)  # (N, K)
    return float(-np.mean(np.sum(p * log_p, axis=-1)))


def fit_gaussian_to_mixture_kl(
    xs: np.ndarray, p_target: np.ndarray, direction: str
) -> tuple[float, float]:
    """Fit a single Gaussian ``q`` to a 1-D target ``p`` on a grid by minimising
    forward ``KL(p || q)`` (mode-covering) or reverse ``KL(q || p)`` (mode-seeking).

    Grid search over (mu, sigma); used for the chapter figure.

    Args:
        xs: (G,) grid. p_target: (G,) normalised so that ``sum(p) = 1``.
        direction: ``"forward"`` or ``"reverse"``.
    Returns:
        (mu, sigma) of the best Gaussian.
    """
    best, best_params = np.inf, (0.0, 1.0)
    for mu in np.linspace(xs.min(), xs.max(), 121):
        for sigma in np.linspace(0.1, 0.5 * (xs.max() - xs.min()), 60):
            q = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)  # (G,)
            q = q / q.sum() + 1e-300
            val = kl_divergence(p_target, q) if direction == "forward" else kl_divergence(q, p_target)
            if val < best:
                best, best_params = float(val), (float(mu), float(sigma))
    return best_params
