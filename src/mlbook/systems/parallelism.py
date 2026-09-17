"""Collective-communication cost models and a parallelism decision procedure.

Ring algorithms on N ranks, tensor of S bytes (per rank), link bandwidth beta
(bytes/s), per-message latency alpha (s):

    reduce-scatter : N-1 steps, each sends S/N   -> bytes on wire per rank (N-1)/N * S
    all-gather     : N-1 steps, each sends S/N   -> (N-1)/N * S
    all-reduce     : reduce-scatter + all-gather -> 2 (N-1)/N * S   (-> 2S as N grows)
    all-to-all     : every rank sends S/N to each of N-1 peers -> (N-1)/N * S
    broadcast      : (N-1)/N * S with a pipelined ring (chunked)

    time = steps * alpha + bytes / beta

Communication per optimizer step for a model with P parameters (gradients in
``g`` bytes each, parameters in ``w`` bytes each), data-parallel degree N:

    DDP      : all-reduce(grads)                         = 2 (N-1)/N * gP
    ZeRO-1/2 : reduce-scatter(grads) + all-gather(params)= (N-1)/N * (gP + wP)
    ZeRO-3   : all-gather(params) fwd + all-gather(params) bwd + reduce-scatter(grads)
             = (N-1)/N * (2 wP + gP)          (1.5x DDP when w == g)
"""

from __future__ import annotations

from dataclasses import dataclass

from mlbook.systems.memory_calc import GB, ParallelPlan, TransformerConfig, training_memory_per_gpu


@dataclass(frozen=True)
class Link:
    """A communication link: ``bandwidth`` in bytes/s per rank, ``latency`` in seconds."""

    name: str
    bandwidth: float
    latency: float = 5e-6


# Order-of-magnitude figures from vendor datasheets (see chapter §4 for sources):
# NVLink 4 on H100: 900 GB/s aggregate per GPU (bidirectional); we model ~450 GB/s
# per direction. InfiniBand NDR: 400 Gb/s = 50 GB/s per port.
NVLINK_H100 = Link("NVLink (H100, per direction)", 450e9, 2e-6)
INFINIBAND_NDR = Link("InfiniBand NDR 400 Gb/s (one port)", 50e9, 5e-6)
PCIE_GEN5_X16 = Link("PCIe 5.0 x16 (per direction)", 64e9, 5e-6)


def ring_bytes_on_wire(op: str, size_bytes: float, n: int) -> float:
    """Bytes each rank sends for ring ``op`` on a ``size_bytes`` tensor over ``n`` ranks."""
    if n <= 1:
        return 0.0
    frac = (n - 1) / n
    return {
        "reduce_scatter": frac * size_bytes,
        "all_gather": frac * size_bytes,
        "all_reduce": 2.0 * frac * size_bytes,
        "all_to_all": frac * size_bytes,
        "broadcast": frac * size_bytes,
    }[op]


def ring_steps(op: str, n: int) -> int:
    """Number of sequential communication rounds for ring ``op``."""
    if n <= 1:
        return 0
    return {"reduce_scatter": n - 1, "all_gather": n - 1, "all_reduce": 2 * (n - 1),
            "all_to_all": n - 1, "broadcast": n - 1}[op]


def collective_time(op: str, size_bytes: float, n: int, link: Link) -> float:
    """Alpha-beta model: steps * latency + bytes_on_wire / bandwidth."""
    return ring_steps(op, n) * link.latency + ring_bytes_on_wire(op, size_bytes, n) / link.bandwidth


def dp_step_comm_bytes(n_params: float, dp: int, zero_stage: int, grad_bytes: float = 2.0, weight_bytes: float = 2.0) -> float:
    """Bytes each rank sends per optimizer step under DDP / ZeRO-1 / -2 / -3."""
    gP, wP = grad_bytes * n_params, weight_bytes * n_params
    if zero_stage == 0:
        return ring_bytes_on_wire("all_reduce", gP, dp)
    if zero_stage in (1, 2):
        return ring_bytes_on_wire("reduce_scatter", gP, dp) + ring_bytes_on_wire("all_gather", wP, dp)
    return 2.0 * ring_bytes_on_wire("all_gather", wP, dp) + ring_bytes_on_wire("reduce_scatter", gP, dp)


def tp_comm_bytes_per_layer(s: int, b: int, h: int, tp: int, act_bytes: float = 2.0) -> float:
    """Tensor-parallel traffic per layer per micro-batch: 2 all-reduces forward
    (attention out-proj, MLP down-proj) + 2 in backward, each on an (s, b, h) tensor."""
    return 4.0 * ring_bytes_on_wire("all_reduce", s * b * h * act_bytes, tp)


def pp_comm_bytes_per_microbatch(s: int, b: int, h: int, act_bytes: float = 2.0) -> float:
    """Point-to-point activation (fwd) + gradient (bwd) across one stage boundary."""
    return 2.0 * s * b * h * act_bytes


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


@dataclass(frozen=True)
class PlanChoice:
    plan: ParallelPlan
    memory_gb: float
    fits: bool
    reason: str


def choose_parallelism(
    cfg: TransformerConfig,
    n_gpus: int,
    gpu_memory_gb: float,
    seq_len: int,
    micro_batch: int = 1,
    nvlink_domain: int = 8,
    headroom: float = 0.85,
    zero_stage: int = 1,
) -> PlanChoice:
    """Decision procedure for (tp, pp, dp).

    1. Keep tensor parallelism inside the NVLink domain (its all-reduces sit on the
       critical path of every layer) and no larger than the head count.
    2. Prefer the smallest ``tp * pp`` that fits, because every unit of model
       parallelism removes a unit of data parallelism (throughput) and pipeline
       parallelism adds bubble.
    3. Among ties prefer larger ``tp`` up to the NVLink domain (bubble-free) over ``pp``.
    """
    budget = gpu_memory_gb * headroom
    best: PlanChoice | None = None
    for tp in [d for d in _divisors(nvlink_domain) if cfg.n_heads % d == 0]:
        for pp in [d for d in _divisors(n_gpus // tp) if cfg.n_layers % d == 0]:
            dp = n_gpus // (tp * pp)
            plan = ParallelPlan(dp=dp, tp=tp, pp=pp, zero_stage=zero_stage)
            mem = training_memory_per_gpu(cfg, plan, seq_len, micro_batch, recompute="selective",
                                          sequence_parallel=True).total / GB
            fits = mem <= budget
            cand = PlanChoice(plan, mem, fits, "")
            if best is None or _better(cand, best):
                best = cand
    assert best is not None
    reason = ("fits with headroom" if best.fits else "does not fit: add GPUs, ZeRO-3, or recompute")
    return PlanChoice(best.plan, best.memory_gb, best.fits, reason)


def _better(a: PlanChoice, b: PlanChoice) -> bool:
    """Prefer fitting plans, then fewer model-parallel ranks, then larger tp, then less memory."""
    ka = (not a.fits, a.plan.tp * a.plan.pp, -a.plan.tp, a.memory_gb)
    kb = (not b.fits, b.plan.tp * b.plan.pp, -b.plan.tp, b.memory_gb)
    return ka < kb
