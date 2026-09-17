"""Megatron-style tensor parallelism simulated with explicit shards in one process.

Row-major convention: X in R^{N x d_in}, a linear layer is Y = X W with W in
R^{d_in x d_out}.

Column-parallel Linear (shard W along d_out):      W = [W_1 | W_2 | ... | W_t]
    Y_i = X W_i                                     (N, d_out / t) on rank i
    no communication in forward (X is replicated); the output stays sharded.

Row-parallel Linear (shard W along d_in):           W = [W_1 ; W_2 ; ... ; W_t]
    input already sharded along d_in: X_i = (N, d_in / t)
    Y = sum_i X_i W_i                               partial sums -> ONE all-reduce.

MLP block:  Y = GELU(X A) B  with A column-sharded and B row-sharded.
    GELU is element-wise, so GELU(X A)_i = GELU(X A_i): the column shards feed the
    row shards with no communication. One all-reduce (after B) in forward, one in
    backward (the gradient w.r.t. X of the column-parallel layer sums over shards).

Attention: heads are split across ranks (W_q, W_k, W_v column-sharded by head), the
output projection W_o is row-sharded -> again exactly one all-reduce in forward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def simulated_all_reduce(partials: list[torch.Tensor]) -> torch.Tensor:
    """Sum of per-rank partial results (what NCCL all-reduce returns on every rank)."""
    out = partials[0].clone()  # (N, d_out)
    for p in partials[1:]:
        out = out + p  # (N, d_out)
    return out


def column_shards(W: torch.Tensor, tp: int) -> list[torch.Tensor]:
    """Split W (d_in, d_out) into tp shards of shape (d_in, d_out / tp)."""
    return list(torch.chunk(W, tp, dim=1))


def row_shards(W: torch.Tensor, tp: int) -> list[torch.Tensor]:
    """Split W (d_in, d_out) into tp shards of shape (d_in / tp, d_out)."""
    return list(torch.chunk(W, tp, dim=0))


def column_parallel_forward(X: torch.Tensor, W_shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Y_i = X W_i on each rank. X: (N, d_in) replicated; returns tp tensors (N, d_out/tp)."""
    return [X @ W_i for W_i in W_shards]  # each (N, d_out / tp)


def row_parallel_forward(X_shards: list[torch.Tensor], W_shards: list[torch.Tensor]) -> torch.Tensor:
    """Y = all_reduce_i(X_i W_i). X_i: (N, d_in/tp), W_i: (d_in/tp, d_out) -> (N, d_out)."""
    partials = [X_i @ W_i for X_i, W_i in zip(X_shards, W_shards)]  # each (N, d_out)
    return simulated_all_reduce(partials)


def tensor_parallel_mlp(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, tp: int) -> torch.Tensor:
    """GELU(X A) B with A column-sharded and B row-sharded; one all-reduce total.

    X: (N, h), A: (h, d_ff), B: (d_ff, h) -> (N, h).
    """
    A_shards = column_shards(A, tp)  # tp x (h, d_ff / tp)
    B_shards = row_shards(B, tp)  # tp x (d_ff / tp, h)
    hidden_shards = [F.gelu(H_i) for H_i in column_parallel_forward(X, A_shards)]  # tp x (N, d_ff / tp)
    return row_parallel_forward(hidden_shards, B_shards)  # (N, h)


def reference_mlp(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Unsharded GELU(X A) B. X: (N, h) -> (N, h)."""
    return F.gelu(X @ A) @ B


def tensor_parallel_attention(
    X: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    n_heads: int,
    tp: int,
) -> torch.Tensor:
    """Head-parallel self-attention (no mask) with one all-reduce after W_o.

    X: (T, d), W_q/W_k/W_v: (d, d), W_o: (d, d); heads split ``tp`` ways, so each rank
    owns n_heads / tp heads, i.e. d / tp columns of W_q, W_k, W_v and d / tp rows of W_o.
    Returns (T, d).
    """
    T, d = X.shape
    d_head = d // n_heads
    heads_per_rank = n_heads // tp
    partials: list[torch.Tensor] = []
    for Wq_i, Wk_i, Wv_i, Wo_i in zip(column_shards(W_q, tp), column_shards(W_k, tp),
                                      column_shards(W_v, tp), row_shards(W_o, tp)):
        Q = (X @ Wq_i).view(T, heads_per_rank, d_head).transpose(0, 1)  # (H_i, T, d_head)
        K = (X @ Wk_i).view(T, heads_per_rank, d_head).transpose(0, 1)  # (H_i, T, d_head)
        V = (X @ Wv_i).view(T, heads_per_rank, d_head).transpose(0, 1)  # (H_i, T, d_head)
        scores = Q @ K.transpose(1, 2) / d_head**0.5  # (H_i, T, T)
        ctx = torch.softmax(scores, dim=-1) @ V  # (H_i, T, d_head)
        ctx = ctx.transpose(0, 1).reshape(T, heads_per_rank * d_head)  # (T, d / tp)
        partials.append(ctx @ Wo_i)  # (T, d)  partial sum over this rank's heads
    return simulated_all_reduce(partials)  # (T, d)


def reference_attention(X, W_q, W_k, W_v, W_o, n_heads):
    """Unsharded multi-head attention, same math as ``tensor_parallel_attention`` with tp=1."""
    return tensor_parallel_attention(X, W_q, W_k, W_v, W_o, n_heads, tp=1)
