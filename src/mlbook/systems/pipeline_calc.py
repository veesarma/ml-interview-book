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


def bubble_fraction(p: int, m: int, v: int = 1) -> float:
    """Fraction of wall-clock spent idle: (p-1) / (v m + p - 1)."""
    if p <= 1:
        return 0.0
    return (p - 1) / (v * m + p - 1)


def bubble_over_ideal(p: int, m: int, v: int = 1) -> float:
    """Bubble time relative to the ideal (bubble-free) time: (p-1) / (v m)."""
    return (p - 1) / (v * m)


@dataclass(frozen=True)
class Op:
    stage: int
    micro: int
    kind: str  # "F" or "B"
    start: float
    end: float


def _op_order(schedule: str, p: int, m: int, stage: int) -> list[tuple[str, int]]:
    if schedule == "gpipe":
        return [("F", j) for j in range(m)] + [("B", j) for j in range(m)]
    warm = min(p - 1 - stage, m)  # 1F1B warm-up forwards on this stage
    ops = [("F", j) for j in range(warm)]
    for k in range(m - warm):
        ops += [("F", warm + k), ("B", k)]
    ops += [("B", j) for j in range(m - warm, m)]
    return ops


def simulate_schedule(schedule: str, p: int, m: int, t_f: float = 1.0, t_b: float = 2.0) -> list[Op]:
    """List-schedule ``gpipe`` or ``1f1b`` and return every op with start/end times.

    Dependencies: F(j) on stage i needs F(j) on stage i-1; B(j) on stage i needs B(j)
    on stage i+1 and F(j) on stage i. Each stage runs one op at a time in its fixed order.
    """
    orders = [_op_order(schedule, p, m, i) for i in range(p)]
    heads = [0] * p
    free = [0.0] * p
    done: dict[tuple[str, int, int], float] = {}
    ops: list[Op] = []
    while any(h < len(o) for h, o in zip(heads, orders)):
        progressed = False
        for i in range(p):
            if heads[i] >= len(orders[i]):
                continue
            kind, j = orders[i][heads[i]]
            dep = ("F", j, i - 1) if kind == "F" else ("B", j, i + 1)
            if kind == "F" and i == 0:
                ready = 0.0
            elif kind == "B" and i == p - 1:
                ready = done[("F", j, i)]
            elif dep in done:
                ready = max(done[dep], done.get(("F", j, i), 0.0))
            else:
                continue
            start = max(free[i], ready)
            end = start + (t_f if kind == "F" else t_b)
            done[(kind, j, i)] = end
            free[i] = end
            ops.append(Op(i, j, kind, start, end))
            heads[i] += 1
            progressed = True
        assert progressed, "schedule deadlocked"
    return ops


def simulated_bubble_fraction(ops: list[Op], p: int) -> float:
    """1 - busy / (p * makespan) averaged over stages."""
    makespan = max(o.end for o in ops)
    busy = sum(o.end - o.start for o in ops)
    return 1.0 - busy / (p * makespan)


def peak_in_flight(ops: list[Op], p: int) -> list[int]:
    """Max number of micro-batches whose activations are live on each stage
    (between its F(j) start and B(j) end)."""
    peaks = []
    for i in range(p):
        events = []
        for o in ops:
            if o.stage != i:
                continue
            events.append((o.start, 1) if o.kind == "F" else (o.end, -1))
        events.sort()
        live = peak = 0
        for _, d in events:
            live += d
            peak = max(peak, live)
        peaks.append(peak)
    return peaks
