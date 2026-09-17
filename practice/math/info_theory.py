# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/math/info_theory.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k info_theory -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py math/info_theory --force

"""Entropy, cross-entropy, KL, mutual information, perplexity, InfoNCE (pure NumPy).

All functions take probability vectors ``(K,)`` or batches ``(N, K)`` over the
last axis and work in nats unless ``base`` is given. ``0 log 0`` is treated as 0.
"""
from __future__ import annotations
import numpy as np

def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``x * log(y)`` with the convention ``0 * log(0) = 0`` (elementwise)."""
    raise NotImplementedError('TODO: implement _xlogy (see the reference in src/mlbook)')

def entropy(p: np.ndarray, base: float | None=None) -> np.ndarray:
    """``H(p) = -sum_k p_k log p_k`` over the last axis.

    Args:
        p: (K,) or (N, K) probabilities.
    Returns:
        scalar or (N,). Nats by default; ``base=2`` for bits.
    """
    raise NotImplementedError('TODO: implement entropy (see the reference in src/mlbook)')

def cross_entropy(p: np.ndarray, q: np.ndarray, base: float | None=None) -> np.ndarray:
    """``H(p, q) = -sum_k p_k log q_k`` (expected code length using q's code for p's data).

    Args:
        p: (..., K) true distribution. q: (..., K) model distribution.
    """
    raise NotImplementedError('TODO: implement cross_entropy (see the reference in src/mlbook)')

def kl_divergence(p: np.ndarray, q: np.ndarray, base: float | None=None) -> np.ndarray:
    """``KL(p || q) = sum_k p_k log(p_k / q_k) = H(p, q) - H(p) >= 0``.

    Args:
        p, q: (..., K). Requires ``q_k > 0`` wherever ``p_k > 0``.
    """
    raise NotImplementedError('TODO: implement kl_divergence (see the reference in src/mlbook)')

def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon: ``1/2 KL(p || m) + 1/2 KL(q || m)`` with ``m = (p + q)/2``. Symmetric, bounded by log 2."""
    raise NotImplementedError('TODO: implement js_divergence (see the reference in src/mlbook)')

def conditional_entropy(joint: np.ndarray) -> float:
    """``H(Y | X) = H(X, Y) - H(X)`` from a joint table ``joint[x, y]``.

    Args:
        joint: (Kx, Ky) probabilities summing to one.
    """
    raise NotImplementedError('TODO: implement conditional_entropy (see the reference in src/mlbook)')

def mutual_information(joint: np.ndarray) -> float:
    """``I(X; Y) = KL(p(x, y) || p(x) p(y)) = H(Y) - H(Y | X)``.

    Args:
        joint: (Kx, Ky).
    """
    raise NotImplementedError('TODO: implement mutual_information (see the reference in src/mlbook)')

def gaussian_kl(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> np.ndarray:
    """``KL(N(mu1, var1) || N(mu2, var2))`` for diagonal Gaussians, summed over the last axis.

    ``= 1/2 sum [ log(var2/var1) + (var1 + (mu1 - mu2)^2)/var2 - 1 ]``  (the VAE term).
    Args:
        all (..., d).
    """
    raise NotImplementedError('TODO: implement gaussian_kl (see the reference in src/mlbook)')

def perplexity(nll_per_token_nats: np.ndarray) -> float:
    """``PPL = exp(mean token NLL)`` -- the effective branching factor.

    Args:
        nll_per_token_nats: (N_tokens,) negative log-likelihoods in nats.
    """
    raise NotImplementedError('TODO: implement perplexity (see the reference in src/mlbook)')

def bits_per_byte(nll_per_token_nats: np.ndarray, n_bytes: int) -> float:
    """``BPB = (sum of token NLL in bits) / n_bytes`` -- tokenizer-independent.

    Args:
        nll_per_token_nats: (N_tokens,). n_bytes: UTF-8 length of the same text.
    """
    raise NotImplementedError('TODO: implement bits_per_byte (see the reference in src/mlbook)')

def infonce_loss(z_a: np.ndarray, z_b: np.ndarray, temperature: float=0.07) -> float:
    """Symmetric InfoNCE (the CLIP objective) on L2-normalised embeddings.

    ``L = 1/2 [CE(rows of S/tau, diag) + CE(cols of S/tau, diag)]`` with
    ``S = z_a z_b^T``. Minimising it maximises a lower bound on ``I(a; b)``:
    ``I(a; b) >= log N - L``.

    Args:
        z_a: (N, d) e.g. image embeddings. z_b: (N, d) matching text embeddings.
    """
    raise NotImplementedError('TODO: implement infonce_loss (see the reference in src/mlbook)')

def _logsumexp(x: np.ndarray, axis: int, keepdims: bool=False) -> np.ndarray:
    raise NotImplementedError('TODO: implement _logsumexp (see the reference in src/mlbook)')

def entropy_bonus(logits: np.ndarray) -> float:
    """Mean policy entropy from logits ``(N, K)``; added to the PPO objective as ``+ beta H``."""
    raise NotImplementedError('TODO: implement entropy_bonus (see the reference in src/mlbook)')

def fit_gaussian_to_mixture_kl(xs: np.ndarray, p_target: np.ndarray, direction: str) -> tuple[float, float]:
    """Fit a single Gaussian ``q`` to a 1-D target ``p`` on a grid by minimising
    forward ``KL(p || q)`` (mode-covering) or reverse ``KL(q || p)`` (mode-seeking).

    Grid search over (mu, sigma); used for the chapter figure.

    Args:
        xs: (G,) grid. p_target: (G,) normalised so that ``sum(p) = 1``.
        direction: ``"forward"`` or ``"reverse"``.
    Returns:
        (mu, sigma) of the best Gaussian.
    """
    raise NotImplementedError('TODO: implement fit_gaussian_to_mixture_kl (see the reference in src/mlbook)')
