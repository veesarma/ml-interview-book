# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/pipeline_calc.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pipeline_calc -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/pipeline_calc --force

"""Pipeline-parallel schedules: bubble fraction and a tiny schedule simulator.

p stages, m micro-batches, forward time t_f and backward time t_b per micro-batch
per stage. The pipeline is a p-stage assembly line; the first micro-batch takes
p-1 stage-times to reach the last stage, and that ramp-up (plus the matching
ramp-down of the backward pass) is idle time on every stage:

    ideal time  = m (t_f + t_b)
    bubble time = (p - 1)(t_f + t_b)
    bubble fraction of total = (p - 1) / (m + p - 1)        (GPipe and 1F1B alike)
    bubble / ideal            = (p - 1) / m

Interleaved 1F1B with v model chunks per stage (Megatron-LM, Narayanan et al. 2021)
shrinks each stage-time by v: bubble / ideal = (p - 1) / (v m), at v x the
point-to-point communication.

1F1B does not shrink the bubble; it bounds the in-flight micro-batches per stage to
at most p (stage i holds p - i), so activation memory is O(p) instead of O(m).
"""
from __future__ import annotations
from dataclasses import dataclass

def bubble_fraction(p: int, m: int, v: int=1) -> float:
    """Fraction of wall-clock spent idle: (p-1) / (v m + p - 1)."""
    raise NotImplementedError('TODO: implement bubble_fraction (see the reference in src/mlbook)')

def bubble_over_ideal(p: int, m: int, v: int=1) -> float:
    """Bubble time relative to the ideal (bubble-free) time: (p-1) / (v m)."""
    raise NotImplementedError('TODO: implement bubble_over_ideal (see the reference in src/mlbook)')

@dataclass(frozen=True)
class Op:
    stage: int
    micro: int
    kind: str
    start: float
    end: float

def _op_order(schedule: str, p: int, m: int, stage: int) -> list[tuple[str, int]]:
    raise NotImplementedError('TODO: implement _op_order (see the reference in src/mlbook)')

def simulate_schedule(schedule: str, p: int, m: int, t_f: float=1.0, t_b: float=2.0) -> list[Op]:
    """List-schedule ``gpipe`` or ``1f1b`` and return every op with start/end times.

    Dependencies: F(j) on stage i needs F(j) on stage i-1; B(j) on stage i needs B(j)
    on stage i+1 and F(j) on stage i. Each stage runs one op at a time in its fixed order.
    """
    raise NotImplementedError('TODO: implement simulate_schedule (see the reference in src/mlbook)')

def simulated_bubble_fraction(ops: list[Op], p: int) -> float:
    """1 - busy / (p * makespan) averaged over stages."""
    raise NotImplementedError('TODO: implement simulated_bubble_fraction (see the reference in src/mlbook)')

def peak_in_flight(ops: list[Op], p: int) -> list[int]:
    """Max number of micro-batches whose activations are live on each stage
    (between its F(j) start and B(j) end)."""
    raise NotImplementedError('TODO: implement peak_in_flight (see the reference in src/mlbook)')
