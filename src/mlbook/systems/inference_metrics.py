"""Prefill vs decode arithmetic intensity, latency/throughput estimates, capacity planning.

Weights: P parameters in ``w`` bytes each (2 for bf16). A GPU has peak compute
``F`` (FLOP/s) and HBM bandwidth ``BW`` (B/s); its ridge point is F / BW FLOP/byte.

Prefill of a prompt with S tokens (batch of 1):
    FLOPs = 2 P S,  bytes >= w P  ->  intensity ~ 2 S / w FLOP/byte  (compute-bound for S >~ 300)

Decode step for a batch of B sequences at context length T:
    FLOPs = 2 P B,  bytes = w P + B * T * kv_bytes_per_token
    intensity (ignoring KV) = 2 B / w  ->  memory-bound until B* = ridge * w / 2
    e.g. H100: ridge = 989e12 / 3.35e12 ~ 295 FLOP/B  ->  B* ~ 295 sequences (bf16).

Time per decode step = max(compute time, memory time) (the roofline bound).
TTFT ~ prefill time; TPOT ~ decode-step time; throughput = B / TPOT.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlbook.systems.memory_calc import BYTES_PER_DTYPE, TransformerConfig, count_params, kv_cache_bytes_per_token


@dataclass(frozen=True)
class GPU:
    name: str
    peak_flops: float  # dense BF16 FLOP/s
    hbm_bandwidth: float  # bytes/s
    memory_bytes: float
    hourly_cost_usd: float = 0.0

    @property
    def ridge_point(self) -> float:
        return self.peak_flops / self.hbm_bandwidth


# Datasheet figures (NVIDIA H100 SXM, A100 SXM 80 GB); cost left at 0 (fill in yours).
H100_SXM = GPU("H100 SXM", 989e12, 3.35e12, 80 * 1024**3)
A100_80GB = GPU("A100 SXM 80GB", 312e12, 2.0e12, 80 * 1024**3)


def prefill_intensity(prompt_len: int, weight_dtype: str = "bf16") -> float:
    """~ 2 S / w FLOP per byte of weights read."""
    return 2.0 * prompt_len / BYTES_PER_DTYPE[weight_dtype]


def decode_intensity(batch: int, weight_dtype: str = "bf16") -> float:
    """~ 2 B / w FLOP per byte of weights read (KV traffic ignored)."""
    return 2.0 * batch / BYTES_PER_DTYPE[weight_dtype]


def decode_ridge_batch(gpu: GPU, weight_dtype: str = "bf16") -> float:
    """Batch size at which a decode step becomes compute-bound: B* = ridge * w / 2."""
    return gpu.ridge_point * BYTES_PER_DTYPE[weight_dtype] / 2.0


@dataclass(frozen=True)
class LatencyEstimate:
    ttft_s: float
    tpot_s: float
    tokens_per_s: float
    e2e_s: float
    kv_bytes: float

    def cost_per_million_tokens(self, gpu: GPU) -> float:
        return gpu.hourly_cost_usd / 3600.0 / self.tokens_per_s * 1e6


def estimate_latency(
    cfg: TransformerConfig,
    gpu: GPU,
    batch: int,
    prompt_len: int,
    gen_len: int,
    weight_dtype: str = "bf16",
    kv_dtype: str = "bf16",
    n_gpus: int = 1,
    efficiency: float = 0.7,
) -> LatencyEstimate:
    """Roofline estimate of TTFT, TPOT, throughput for one replica of ``n_gpus``.

    ``efficiency`` scales both peak numbers (kernels do not hit datasheet peaks).
    Weights and KV are assumed to be spread evenly across ``n_gpus`` (tensor parallel).
    """
    P = count_params(cfg)["total"]
    w = BYTES_PER_DTYPE[weight_dtype]
    flops = gpu.peak_flops * n_gpus * efficiency
    bw = gpu.hbm_bandwidth * n_gpus * efficiency
    kv_tok = kv_cache_bytes_per_token(cfg, kv_dtype)

    prefill_flops = 2.0 * P * prompt_len * batch
    prefill_bytes = w * P + batch * prompt_len * kv_tok
    ttft = max(prefill_flops / flops, prefill_bytes / bw)

    mid_ctx = prompt_len + gen_len / 2.0
    step_flops = 2.0 * P * batch
    step_bytes = w * P + batch * mid_ctx * kv_tok
    tpot = max(step_flops / flops, step_bytes / bw)

    kv_total = batch * (prompt_len + gen_len) * kv_tok
    return LatencyEstimate(ttft, tpot, batch / tpot, ttft + gen_len * tpot, kv_total)


def max_batch_for_memory(cfg: TransformerConfig, gpu: GPU, context_len: int, weight_dtype="bf16", kv_dtype="bf16", n_gpus: int = 1) -> int:
    """Largest batch whose KV cache fits next to the weights (ignores workspace)."""
    P = count_params(cfg)["total"]
    free = gpu.memory_bytes * n_gpus - BYTES_PER_DTYPE[weight_dtype] * P
    per_seq = context_len * kv_cache_bytes_per_token(cfg, kv_dtype)
    return max(0, int(free // per_seq))


def replicas_for_slo(
    cfg: TransformerConfig,
    gpu: GPU,
    requests_per_s: float,
    prompt_len: int,
    gen_len: int,
    tpot_slo_s: float,
    n_gpus_per_replica: int = 1,
    efficiency: float = 0.7,
) -> dict[str, float]:
    """Capacity plan: largest batch meeting the TPOT SLO, then replicas = load / capacity."""
    best_batch, best_tps = 0, 0.0
    b_mem = max_batch_for_memory(cfg, gpu, prompt_len + gen_len, n_gpus=n_gpus_per_replica)
    for b in range(1, b_mem + 1):
        est = estimate_latency(cfg, gpu, b, prompt_len, gen_len, n_gpus=n_gpus_per_replica, efficiency=efficiency)
        if est.tpot_s > tpot_slo_s:
            break
        best_batch, best_tps = b, est.tokens_per_s
    if best_batch == 0:
        return {"batch": 0, "replicas": float("inf"), "tokens_per_s_per_replica": 0.0}
    demand_tps = requests_per_s * gen_len
    import math

    return {"batch": best_batch, "tokens_per_s_per_replica": best_tps,
            "replicas": math.ceil(demand_tps / best_tps), "max_batch_by_memory": b_mem}
