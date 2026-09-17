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
    t_fixed: float = 0.005  # per-iteration overhead (kernel launches, scheduler)
    c_prefill: float = 2e-5  # seconds per prompt token
    c_decode: float = 5e-4  # seconds per decoding sequence per step


@dataclass
class Metrics:
    makespan: float
    output_tokens: int
    completed: list[Completed] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.output_tokens / self.makespan

    @property
    def mean_latency(self) -> float:
        return sum(c.finish - c.arrival for c in self.completed) / len(self.completed)

    @property
    def mean_ttft(self) -> float:
        return sum(c.first_token - c.arrival for c in self.completed) / len(self.completed)


def simulate_static(requests: list[Request], max_batch: int, cost: CostModel = CostModel()) -> Metrics:
    """Batches run to completion; a slot is not reused until the whole batch finishes."""
    pending = sorted(requests, key=lambda r: r.arrival)
    t, done, tokens = 0.0, [], 0
    while pending:
        t = max(t, pending[0].arrival)
        batch = [r for r in pending if r.arrival <= t][:max_batch]
        pending = [r for r in pending if r not in batch]
        t += cost.t_fixed + cost.c_prefill * sum(r.prompt_len for r in batch)  # prefill iteration
        first = t
        steps = max(r.gen_len for r in batch)
        for step in range(1, steps):  # first token came from prefill; remaining decode steps
            t += cost.t_fixed + cost.c_decode * len(batch)  # padding: all slots run
        for r in batch:
            finish = first + sum(cost.t_fixed + cost.c_decode * len(batch) for _ in range(r.gen_len - 1))
            done.append(Completed(r.rid, r.arrival, first, finish, r.gen_len))
            tokens += r.gen_len
    return Metrics(t, tokens, done)


def simulate_continuous(requests: list[Request], max_batch: int, cost: CostModel = CostModel()) -> Metrics:
    """Iteration-level scheduling: finished sequences leave, new ones join each step."""
    pending = sorted(requests, key=lambda r: r.arrival)
    running: list[tuple[Request, int]] = []  # (request, tokens generated so far)
    first_token: dict[int, float] = {}
    t, done, tokens = 0.0, [], 0
    while pending or running:
        if not running and pending:
            t = max(t, pending[0].arrival)
        admitted = []
        while pending and pending[0].arrival <= t and len(running) < max_batch:
            admitted.append(pending.pop(0))
        running += [(r, 0) for r in admitted]
        t += cost.t_fixed + cost.c_prefill * sum(r.prompt_len for r in admitted) + cost.c_decode * (len(running) - len(admitted))
        still = []
        for r, n in running:
            n += 1
            if n == 1:
                first_token[r.rid] = t
            if n >= r.gen_len:
                done.append(Completed(r.rid, r.arrival, first_token[r.rid], t, r.gen_len))
                tokens += r.gen_len
            else:
                still.append((r, n))
        running = still
    return Metrics(t, tokens, done)


def make_workload(n: int, rate: float, seed: int = 0, mean_gen: int = 64) -> list[Request]:
    """Poisson arrivals at ``rate`` req/s, geometric-ish generation lengths."""
    import numpy as np

    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=n))  # (n,)
    gens = np.maximum(1, rng.geometric(1.0 / mean_gen, size=n))  # (n,)
    prompts = rng.integers(32, 256, size=n)  # (n,)
    return [Request(i, float(arrivals[i]), int(prompts[i]), int(gens[i])) for i in range(n)]
