# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/tensor_parallel_toy.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tensor_parallel_toy -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/tensor_parallel_toy --force

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
    raise NotImplementedError('TODO: implement simulated_all_reduce (see the reference in src/mlbook)')

def column_shards(W: torch.Tensor, tp: int) -> list[torch.Tensor]:
    """Split W (d_in, d_out) into tp shards of shape (d_in, d_out / tp)."""
    raise NotImplementedError('TODO: implement column_shards (see the reference in src/mlbook)')

def row_shards(W: torch.Tensor, tp: int) -> list[torch.Tensor]:
    """Split W (d_in, d_out) into tp shards of shape (d_in / tp, d_out)."""
    raise NotImplementedError('TODO: implement row_shards (see the reference in src/mlbook)')

def column_parallel_forward(X: torch.Tensor, W_shards: list[torch.Tensor]) -> list[torch.Tensor]:
    """Y_i = X W_i on each rank. X: (N, d_in) replicated; returns tp tensors (N, d_out/tp)."""
    raise NotImplementedError('TODO: implement column_parallel_forward (see the reference in src/mlbook)')

def row_parallel_forward(X_shards: list[torch.Tensor], W_shards: list[torch.Tensor]) -> torch.Tensor:
    """Y = all_reduce_i(X_i W_i). X_i: (N, d_in/tp), W_i: (d_in/tp, d_out) -> (N, d_out)."""
    raise NotImplementedError('TODO: implement row_parallel_forward (see the reference in src/mlbook)')

def tensor_parallel_mlp(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, tp: int) -> torch.Tensor:
    """GELU(X A) B with A column-sharded and B row-sharded; one all-reduce total.

    X: (N, h), A: (h, d_ff), B: (d_ff, h) -> (N, h).
    """
    raise NotImplementedError('TODO: implement tensor_parallel_mlp (see the reference in src/mlbook)')

def reference_mlp(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Unsharded GELU(X A) B. X: (N, h) -> (N, h)."""
    raise NotImplementedError('TODO: implement reference_mlp (see the reference in src/mlbook)')

def tensor_parallel_attention(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_o: torch.Tensor, n_heads: int, tp: int) -> torch.Tensor:
    """Head-parallel self-attention (no mask) with one all-reduce after W_o.

    X: (T, d), W_q/W_k/W_v: (d, d), W_o: (d, d); heads split ``tp`` ways, so each rank
    owns n_heads / tp heads, i.e. d / tp columns of W_q, W_k, W_v and d / tp rows of W_o.
    Returns (T, d).
    """
    raise NotImplementedError('TODO: implement tensor_parallel_attention (see the reference in src/mlbook)')

def reference_attention(X, W_q, W_k, W_v, W_o, n_heads):
    """Unsharded multi-head attention, same math as ``tensor_parallel_attention`` with tp=1."""
    raise NotImplementedError('TODO: implement reference_attention (see the reference in src/mlbook)')
