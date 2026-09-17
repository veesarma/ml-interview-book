# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/continuous_batching_sim.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k continuous_batching_sim -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/continuous_batching_sim --force

"""Static vs continuous (iteration-level) batching, as a discrete-event simulation.

Cost model per iteration (one forward pass of the batch):
    t_iter = t_fixed + c_prefill * (prompt tokens processed) + c_decode * (sequences decoding)

Static batching (naive): form a batch of up to B requests, run it to completion
(all sequences generate until the *longest* is done; finished ones are padding),
then admit the next batch.

Continuous batching (Orca, Yu et al. 2022; vLLM): after every iteration, evict
finished sequences and admit waiting ones into the freed slots. Prefill of a new
request is merged into the iteration (as in chunked prefill / vLLM's scheduler).
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Request:
    rid: int
    arrival: float
    prompt_len: int
    gen_len: int

@dataclass
class Completed:
    rid: int
    arrival: float
    first_token: float
    finish: float
    gen_len: int

@dataclass(frozen=True)
class CostModel:
    t_fixed: float = 0.005
    c_prefill: float = 2e-05
    c_decode: float = 0.0005

@dataclass
class Metrics:
    makespan: float
    output_tokens: int
    completed: list[Completed] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        raise NotImplementedError('TODO: implement throughput (see the reference in src/mlbook)')

    @property
    def mean_latency(self) -> float:
        raise NotImplementedError('TODO: implement mean_latency (see the reference in src/mlbook)')

    @property
    def mean_ttft(self) -> float:
        raise NotImplementedError('TODO: implement mean_ttft (see the reference in src/mlbook)')

def simulate_static(requests: list[Request], max_batch: int, cost: CostModel=CostModel()) -> Metrics:
    """Batches run to completion; a slot is not reused until the whole batch finishes."""
    raise NotImplementedError('TODO: implement simulate_static (see the reference in src/mlbook)')

def simulate_continuous(requests: list[Request], max_batch: int, cost: CostModel=CostModel()) -> Metrics:
    """Iteration-level scheduling: finished sequences leave, new ones join each step."""
    raise NotImplementedError('TODO: implement simulate_continuous (see the reference in src/mlbook)')

def make_workload(n: int, rate: float, seed: int=0, mean_gen: int=64) -> list[Request]:
    """Poisson arrivals at ``rate`` req/s, geometric-ish generation lengths."""
    raise NotImplementedError('TODO: implement make_workload (see the reference in src/mlbook)')
