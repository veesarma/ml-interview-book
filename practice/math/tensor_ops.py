# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/math/tensor_ops.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tensor_ops -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py math/tensor_ops --force

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
    raise NotImplementedError('TODO: implement split_heads (see the reference in src/mlbook)')

def merge_heads(x: np.ndarray) -> np.ndarray:
    """(B, H, T, d_head) -> (B, T, d_model): inverse of ``split_heads``.

    The transpose makes the array non-contiguous, so the final reshape has to
    copy (in PyTorch: ``.transpose(1, 2).contiguous().view(B, T, -1)``).
    """
    raise NotImplementedError('TODO: implement merge_heads (see the reference in src/mlbook)')

def batched_outer(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(B, m) x (B, n) -> (B, m, n) with ``out[b] = u[b] v[b]^T`` via broadcasting."""
    raise NotImplementedError('TODO: implement batched_outer (see the reference in src/mlbook)')

def causal_mask(T: int) -> np.ndarray:
    """(1, 1, T, T) additive mask: 0 where key j <= query i, -inf above the diagonal.

    Broadcasts against scores of shape (B, H, T, T).
    """
    raise NotImplementedError('TODO: implement causal_mask (see the reference in src/mlbook)')

def padding_mask(lengths: np.ndarray, T: int) -> np.ndarray:
    """(B,) valid lengths -> (B, 1, 1, T) additive mask: 0 for real tokens, -inf for pads."""
    raise NotImplementedError('TODO: implement padding_mask (see the reference in src/mlbook)')

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Multi-head attention with an additive mask, all shapes batched.

    Args:
        Q, K: (B, H, T, d_head). V: (B, H, T, d_head).
        mask: broadcastable to (B, H, T, T), entries 0 or -inf.
    Returns:
        (B, H, T, d_head).
    """
    raise NotImplementedError('TODO: implement masked_attention (see the reference in src/mlbook)')

def gather_token_logprobs(log_probs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Pick the log-prob of the target token at every position.

    Args:
        log_probs: (B, T, V) log-softmax output. targets: (B, T) integer ids.
    Returns:
        (B, T): ``out[b, t] = log_probs[b, t, targets[b, t]]``.
    """
    raise NotImplementedError('TODO: implement gather_token_logprobs (see the reference in src/mlbook)')

def one_hot(targets: np.ndarray, K: int) -> np.ndarray:
    """(N,) ints -> (N, K) one-hot floats via a broadcasted comparison."""
    raise NotImplementedError('TODO: implement one_hot (see the reference in src/mlbook)')

def cross_entropy_from_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean cross-entropy ``-mean_n log softmax(logits)[n, y_n]`` (log-sum-exp form).

    Args:
        logits: (N, K). targets: (N,) ints.
    """
    raise NotImplementedError('TODO: implement cross_entropy_from_logits (see the reference in src/mlbook)')

def attention_einsum(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Same as ``masked_attention`` without a mask, written with einsum for the chapter.

    ``"bhtd,bhsd->bhts"``: for each batch b and head h, contract query t (over
    d) against key s (over d) -> scores indexed (t, s).
    ``"bhts,bhsd->bhtd"``: for each (b, h), average value rows s with weights (t, s).
    """
    raise NotImplementedError('TODO: implement attention_einsum (see the reference in src/mlbook)')
