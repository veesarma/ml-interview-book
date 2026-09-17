"""Shape manipulations every interview expects you to write without thinking
(pure NumPy; the PyTorch versions are one-to-one, see the chapter).

Dimension vocabulary: B batch, T sequence length, d_model width, H heads,
d_head = d_model // H, V vocab, K classes, N examples.
"""

from __future__ import annotations

import numpy as np


def split_heads(x: np.ndarray, H: int) -> np.ndarray:
    """(B, T, d_model) -> (B, H, T, d_head): reshape the last axis, then swap T and H.

    The reshape only *reinterprets* the contiguous last axis as (H, d_head);
    the transpose is what puts heads in front so a batched matmul over the
    leading (B, H) axes computes all heads at once.
    """
    B, T, d_model = x.shape
    d_head = d_model // H
    x = x.reshape(B, T, H, d_head)  # (B, T, H, d_head) -- free view
    return x.transpose(0, 2, 1, 3)  # (B, H, T, d_head) -- strided view


def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B, H, T, d_head) -> (B, T, d_model): inverse of ``split_heads``.

    The transpose makes the array non-contiguous, so the final reshape has to
    copy (in PyTorch: ``.transpose(1, 2).contiguous().view(B, T, -1)``).
    """
    B, H, T, d_head = x.shape
    x = x.transpose(0, 2, 1, 3)  # (B, T, H, d_head)
    return x.reshape(B, T, H * d_head)  # (B, T, d_model) -- copies


def batched_outer(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(B, m) x (B, n) -> (B, m, n) with ``out[b] = u[b] v[b]^T`` via broadcasting."""
    return u[:, :, None] * v[:, None, :]  # (B, m, 1) * (B, 1, n) -> (B, m, n)


def causal_mask(T: int) -> np.ndarray:
    """(1, 1, T, T) additive mask: 0 where key j <= query i, -inf above the diagonal.

    Broadcasts against scores of shape (B, H, T, T).
    """
    allowed = np.tril(np.ones((T, T), dtype=bool))  # (T, T) lower-triangular True
    mask = np.where(allowed, 0.0, -np.inf)  # (T, T)
    return mask[None, None, :, :]  # (1, 1, T, T)


def padding_mask(lengths: np.ndarray, T: int) -> np.ndarray:
    """(B,) valid lengths -> (B, 1, 1, T) additive mask: 0 for real tokens, -inf for pads."""
    positions = np.arange(T)[None, :]  # (1, T)
    valid = positions < lengths[:, None]  # (B, T) True where the key is a real token
    mask = np.where(valid, 0.0, -np.inf)  # (B, T)
    return mask[:, None, None, :]  # (B, 1, 1, T)


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multi-head attention with an additive mask, all shapes batched.

    Args:
        Q, K: (B, H, T, d_head). V: (B, H, T, d_head).
        mask: broadcastable to (B, H, T, T), entries 0 or -inf.
    Returns:
        (B, H, T, d_head).
    """
    d_head = Q.shape[-1]
    S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_head)  # (B,H,T,d) @ (B,H,d,T) -> (B, H, T, T)
    S = S + mask  # (B, H, T, T) broadcasting the mask
    S = S - S.max(axis=-1, keepdims=True)  # (B, H, T, T) stable softmax
    A = np.exp(S)  # (B, H, T, T)
    A = A / A.sum(axis=-1, keepdims=True)  # (B, H, T, T) rows sum to 1
    return A @ V  # (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d_head)


def gather_token_logprobs(log_probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Pick the log-prob of the target token at every position.

    Args:
        log_probs: (B, T, V) log-softmax output. targets: (B, T) integer ids.
    Returns:
        (B, T): ``out[b, t] = log_probs[b, t, targets[b, t]]``.
    """
    idx = targets[:, :, None]  # (B, T, 1) index along the last axis
    return np.take_along_axis(log_probs, idx, axis=-1)[:, :, 0]  # (B, T, 1) -> (B, T)


def one_hot(targets: np.ndarray, K: int) -> np.ndarray:
    """(N,) ints -> (N, K) one-hot floats via a broadcasted comparison."""
    return (targets[:, None] == np.arange(K)[None, :]).astype(np.float64)  # (N, 1) == (1, K) -> (N, K)


def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy ``-mean_n log softmax(logits)[n, y_n]`` (log-sum-exp form).

    Args:
        logits: (N, K). targets: (N,) ints.
    """
    m = logits.max(axis=-1, keepdims=True)  # (N, 1)
    log_z = m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))  # (N, 1) log-sum-exp
    log_probs = logits - log_z  # (N, K)
    picked = np.take_along_axis(log_probs, targets[:, None], axis=-1)  # (N, 1)
    return float(-picked.mean())


def attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Same as ``masked_attention`` without a mask, written with einsum for the chapter.

    ``"bhtd,bhsd->bhts"``: for each batch b and head h, contract query t (over
    d) against key s (over d) -> scores indexed (t, s).
    ``"bhts,bhsd->bhtd"``: for each (b, h), average value rows s with weights (t, s).
    """
    d_head = Q.shape[-1]
    S = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(d_head)  # (B, H, T, T)
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S)
    A = A / A.sum(axis=-1, keepdims=True)  # (B, H, T, T)
    return np.einsum("bhts,bhsd->bhtd", A, V)  # (B, H, T, d_head)
