# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/inference_metrics.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k inference_metrics -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/inference_metrics --force

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
    peak_flops: float
    hbm_bandwidth: float
    memory_bytes: float
    hourly_cost_usd: float = 0.0

    @property
    def ridge_point(self) -> float:
        raise NotImplementedError('TODO: implement ridge_point (see the reference in src/mlbook)')
H100_SXM = GPU('H100 SXM', 989000000000000.0, 3350000000000.0, 80 * 1024 ** 3)
A100_80GB = GPU('A100 SXM 80GB', 312000000000000.0, 2000000000000.0, 80 * 1024 ** 3)

def prefill_intensity(prompt_len: int, weight_dtype: str='bf16') -> float:
    """~ 2 S / w FLOP per byte of weights read."""
    raise NotImplementedError('TODO: implement prefill_intensity (see the reference in src/mlbook)')

def decode_intensity(batch: int, weight_dtype: str='bf16') -> float:
    """~ 2 B / w FLOP per byte of weights read (KV traffic ignored)."""
    raise NotImplementedError('TODO: implement decode_intensity (see the reference in src/mlbook)')

def decode_ridge_batch(gpu: GPU, weight_dtype: str='bf16') -> float:
    """Batch size at which a decode step becomes compute-bound: B* = ridge * w / 2."""
    raise NotImplementedError('TODO: implement decode_ridge_batch (see the reference in src/mlbook)')

@dataclass(frozen=True)
class LatencyEstimate:
    ttft_s: float
    tpot_s: float
    tokens_per_s: float
    e2e_s: float
    kv_bytes: float

    def cost_per_million_tokens(self, gpu: GPU) -> float:
        raise NotImplementedError('TODO: implement cost_per_million_tokens (see the reference in src/mlbook)')

def estimate_latency(cfg: TransformerConfig, gpu: GPU, batch: int, prompt_len: int, gen_len: int, weight_dtype: str='bf16', kv_dtype: str='bf16', n_gpus: int=1, efficiency: float=0.7) -> LatencyEstimate:
    """Roofline estimate of TTFT, TPOT, throughput for one replica of ``n_gpus``.

    ``efficiency`` scales both peak numbers (kernels do not hit datasheet peaks).
    Weights and KV are assumed to be spread evenly across ``n_gpus`` (tensor parallel).
    """
    raise NotImplementedError('TODO: implement estimate_latency (see the reference in src/mlbook)')

def max_batch_for_memory(cfg: TransformerConfig, gpu: GPU, context_len: int, weight_dtype='bf16', kv_dtype='bf16', n_gpus: int=1) -> int:
    """Largest batch whose KV cache fits next to the weights (ignores workspace)."""
    raise NotImplementedError('TODO: implement max_batch_for_memory (see the reference in src/mlbook)')

def replicas_for_slo(cfg: TransformerConfig, gpu: GPU, requests_per_s: float, prompt_len: int, gen_len: int, tpot_slo_s: float, n_gpus_per_replica: int=1, efficiency: float=0.7) -> dict[str, float]:
    """Capacity plan: largest batch meeting the TPOT SLO, then replicas = load / capacity."""
    raise NotImplementedError('TODO: implement replicas_for_slo (see the reference in src/mlbook)')
