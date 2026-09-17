# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/ddp_example.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k ddp_example -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/ddp_example --force

"""A runnable 2-process data-parallel example on CPU (torch.distributed, gloo).

Each rank holds an identical copy of a small model, computes the loss on its own
slice of the global batch, and averages gradients with an all-reduce:

    g = (1 / N) * sum_r g_r        where g_r = grad of the mean loss on rank r's slice.

Because the global loss is the mean over the whole batch, this equals the gradient
of the single-process full-batch loss exactly (up to floating-point summation
order). We check that, and also that ``DistributedDataParallel`` produces the same
gradients as our manual all-reduce.
"""
from __future__ import annotations
import os
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

def make_model(seed: int=0) -> nn.Module:
    raise NotImplementedError('TODO: implement make_model (see the reference in src/mlbook)')

def make_data(seed: int=1, n: int=64) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError('TODO: implement make_data (see the reference in src/mlbook)')

def full_batch_gradients(seed: int=0) -> list[torch.Tensor]:
    """Single-process reference: gradient of the mean loss over all N examples."""
    raise NotImplementedError('TODO: implement full_batch_gradients (see the reference in src/mlbook)')

def manual_all_reduce_grads(model: nn.Module, world_size: int) -> None:
    """In place: p.grad <- mean over ranks of p.grad (one all-reduce per tensor)."""
    raise NotImplementedError('TODO: implement manual_all_reduce_grads (see the reference in src/mlbook)')

def _worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    raise NotImplementedError('TODO: implement _worker (see the reference in src/mlbook)')

def run_ddp_demo(world_size: int=2) -> dict[str, float]:
    """Spawn ``world_size`` CPU processes; return max |grad difference| vs the reference.

    Keys: ``manual_vs_full`` and ``ddp_vs_full`` (both should be ~1e-7) and
    ``rank_agreement`` (all ranks hold identical gradients after the all-reduce).
    """
    raise NotImplementedError('TODO: implement run_ddp_demo (see the reference in src/mlbook)')
