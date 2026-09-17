# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/parallelism.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k parallelism -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/parallelism --force

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
    latency: float = 5e-06
NVLINK_H100 = Link('NVLink (H100, per direction)', 450000000000.0, 2e-06)
INFINIBAND_NDR = Link('InfiniBand NDR 400 Gb/s (one port)', 50000000000.0, 5e-06)
PCIE_GEN5_X16 = Link('PCIe 5.0 x16 (per direction)', 64000000000.0, 5e-06)

def ring_bytes_on_wire(op: str, size_bytes: float, n: int) -> float:
    """Bytes each rank sends for ring ``op`` on a ``size_bytes`` tensor over ``n`` ranks."""
    raise NotImplementedError('TODO: implement ring_bytes_on_wire (see the reference in src/mlbook)')

def ring_steps(op: str, n: int) -> int:
    """Number of sequential communication rounds for ring ``op``."""
    raise NotImplementedError('TODO: implement ring_steps (see the reference in src/mlbook)')

def collective_time(op: str, size_bytes: float, n: int, link: Link) -> float:
    """Alpha-beta model: steps * latency + bytes_on_wire / bandwidth."""
    raise NotImplementedError('TODO: implement collective_time (see the reference in src/mlbook)')

def dp_step_comm_bytes(n_params: float, dp: int, zero_stage: int, grad_bytes: float=2.0, weight_bytes: float=2.0) -> float:
    """Bytes each rank sends per optimizer step under DDP / ZeRO-1 / -2 / -3."""
    raise NotImplementedError('TODO: implement dp_step_comm_bytes (see the reference in src/mlbook)')

def tp_comm_bytes_per_layer(s: int, b: int, h: int, tp: int, act_bytes: float=2.0) -> float:
    """Tensor-parallel traffic per layer per micro-batch: 2 all-reduces forward
    (attention out-proj, MLP down-proj) + 2 in backward, each on an (s, b, h) tensor."""
    raise NotImplementedError('TODO: implement tp_comm_bytes_per_layer (see the reference in src/mlbook)')

def pp_comm_bytes_per_microbatch(s: int, b: int, h: int, act_bytes: float=2.0) -> float:
    """Point-to-point activation (fwd) + gradient (bwd) across one stage boundary."""
    raise NotImplementedError('TODO: implement pp_comm_bytes_per_microbatch (see the reference in src/mlbook)')

def _divisors(n: int) -> list[int]:
    raise NotImplementedError('TODO: implement _divisors (see the reference in src/mlbook)')

@dataclass(frozen=True)
class PlanChoice:
    plan: ParallelPlan
    memory_gb: float
    fits: bool
    reason: str

def choose_parallelism(cfg: TransformerConfig, n_gpus: int, gpu_memory_gb: float, seq_len: int, micro_batch: int=1, nvlink_domain: int=8, headroom: float=0.85, zero_stage: int=1) -> PlanChoice:
    """Decision procedure for (tp, pp, dp).

    1. Keep tensor parallelism inside the NVLink domain (its all-reduces sit on the
       critical path of every layer) and no larger than the head count.
    2. Prefer the smallest ``tp * pp`` that fits, because every unit of model
       parallelism removes a unit of data parallelism (throughput) and pipeline
       parallelism adds bubble.
    3. Among ties prefer larger ``tp`` up to the NVLink domain (bubble-free) over ``pp``.
    """
    raise NotImplementedError('TODO: implement choose_parallelism (see the reference in src/mlbook)')

def _better(a: PlanChoice, b: PlanChoice) -> bool:
    """Prefer fitting plans, then fewer model-parallel ranks, then larger tp, then less memory."""
    raise NotImplementedError('TODO: implement _better (see the reference in src/mlbook)')
