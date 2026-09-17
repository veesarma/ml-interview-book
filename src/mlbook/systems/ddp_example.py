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


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))


def make_data(seed: int = 1, n: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 8, generator=g)  # (N, 8)
    y = torch.randn(n, 1, generator=g)  # (N, 1)
    return X, y


def full_batch_gradients(seed: int = 0) -> list[torch.Tensor]:
    """Single-process reference: gradient of the mean loss over all N examples."""
    model = make_model(seed)
    X, y = make_data()
    loss = nn.functional.mse_loss(model(X), y)
    loss.backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def manual_all_reduce_grads(model: nn.Module, world_size: int) -> None:
    """In place: p.grad <- mean over ranks of p.grad (one all-reduce per tensor)."""
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  # same shape as p
        p.grad.div_(world_size)


def _worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    X, y = make_data()
    n_local = X.shape[0] // world_size
    X_r = X[rank * n_local:(rank + 1) * n_local]  # (N / world, 8)
    y_r = y[rank * n_local:(rank + 1) * n_local]  # (N / world, 1)

    # 1) manual data parallelism: local backward, then all-reduce the gradients.
    model = make_model()
    nn.functional.mse_loss(model(X_r), y_r).backward()
    manual_all_reduce_grads(model, world_size)
    manual = [p.grad.detach().clone() for p in model.parameters()]

    # 2) the same thing with DistributedDataParallel (bucketed all-reduce hooks).
    ddp = nn.parallel.DistributedDataParallel(make_model())
    nn.functional.mse_loss(ddp(X_r), y_r).backward()
    via_ddp = [p.grad.detach().clone() for p in ddp.parameters()]

    torch.save({"manual": manual, "ddp": via_ddp}, os.path.join(out_dir, f"rank{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def run_ddp_demo(world_size: int = 2) -> dict[str, float]:
    """Spawn ``world_size`` CPU processes; return max |grad difference| vs the reference.

    Keys: ``manual_vs_full`` and ``ddp_vs_full`` (both should be ~1e-7) and
    ``rank_agreement`` (all ranks hold identical gradients after the all-reduce).
    """
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init")
        mp.spawn(_worker, args=(world_size, init_file, tmp), nprocs=world_size, join=True)
        ref = full_batch_gradients()
        results = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in range(world_size)]
    diff = lambda a, b: max(float((x - y).abs().max()) for x, y in zip(a, b))  # noqa: E731
    return {
        "manual_vs_full": diff(results[0]["manual"], ref),
        "ddp_vs_full": diff(results[0]["ddp"], ref),
        "rank_agreement": max(diff(results[0]["manual"], r["manual"]) for r in results[1:]),
    }


if __name__ == "__main__":
    print(run_ddp_demo())
