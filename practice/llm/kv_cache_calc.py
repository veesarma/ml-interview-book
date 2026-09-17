# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/kv_cache_calc.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k kv_cache_calc -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/kv_cache_calc --force

"""KV-cache and decode arithmetic-intensity calculators (pure Python/NumPy).

Per token, per layer, the cache stores one key and one value vector per KV head:

    bytes_per_token = 2 (K and V) · L · H_kv · d_head · bytes_per_element
    KV cache (batch B, context T) = B · T · bytes_per_token

Multi-head latent attention (DeepSeek-V2/V3) stores a compressed latent
``c_t ∈ R^{d_c}`` plus a decoupled RoPE key ``k^R_t ∈ R^{d_h^R}`` instead of K and V:

    bytes_per_token_MLA = L · (d_c + d_h^R) · bytes_per_element
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelSpec:
    """Attention geometry of a decoder-only Transformer."""
    name: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    n_params: float

    @property
    def d_model(self) -> int:
        raise NotImplementedError('TODO: implement d_model (see the reference in src/mlbook)')
LLAMA2_7B = ModelSpec('Llama-2-7B', n_layers=32, n_heads=32, n_kv_heads=32, d_head=128, n_params=6700000000.0)
LLAMA2_70B = ModelSpec('Llama-2-70B', n_layers=80, n_heads=64, n_kv_heads=8, d_head=128, n_params=69000000000.0)
LLAMA3_8B = ModelSpec('Llama-3-8B', n_layers=32, n_heads=32, n_kv_heads=8, d_head=128, n_params=8000000000.0)
LLAMA3_70B = ModelSpec('Llama-3-70B', n_layers=80, n_heads=64, n_kv_heads=8, d_head=128, n_params=70000000000.0)
LLAMA3_405B = ModelSpec('Llama-3-405B', n_layers=126, n_heads=128, n_kv_heads=8, d_head=128, n_params=405000000000.0)
MISTRAL_7B = ModelSpec('Mistral-7B', n_layers=32, n_heads=32, n_kv_heads=8, d_head=128, n_params=7200000000.0)
LLAMA3_70B_MHA = ModelSpec('Llama-3-70B (if MHA)', n_layers=80, n_heads=64, n_kv_heads=64, d_head=128, n_params=70000000000.0)

def kv_bytes_per_token(spec: ModelSpec, dtype_bytes: int=2) -> int:
    """2 · L · H_kv · d_head · bytes."""
    raise NotImplementedError('TODO: implement kv_bytes_per_token (see the reference in src/mlbook)')

def kv_cache_bytes(spec: ModelSpec, seq_len: int, batch: int=1, dtype_bytes: int=2) -> int:
    """Total KV-cache bytes for ``batch`` sequences of ``seq_len`` tokens."""
    raise NotImplementedError('TODO: implement kv_cache_bytes (see the reference in src/mlbook)')

def mla_bytes_per_token(n_layers: int, d_c: int=512, d_rope: int=64, dtype_bytes: int=2) -> int:
    """MLA cache: L · (d_c + d_h^R) · bytes (DeepSeek-V2 defaults d_c=512, d_h^R=64)."""
    raise NotImplementedError('TODO: implement mla_bytes_per_token (see the reference in src/mlbook)')

def max_batch_at_context(spec: ModelSpec, seq_len: int, hbm_bytes: float, dtype_bytes: int=2) -> int:
    """How many ``seq_len`` sequences fit after the weights are resident in ``hbm_bytes``."""
    raise NotImplementedError('TODO: implement max_batch_at_context (see the reference in src/mlbook)')

def decode_arithmetic_intensity(spec: ModelSpec, batch: int, context: int, dtype_bytes: int=2) -> float:
    """FLOPs per HBM byte for one decode step of a batch (weights read once, KV read per sequence).

    FLOPs  ≈ 2 · N · batch        (one token per sequence through all weights)
    bytes  ≈ N · dtype + batch · context · kv_bytes_per_token
    """
    raise NotImplementedError('TODO: implement decode_arithmetic_intensity (see the reference in src/mlbook)')

def prefill_arithmetic_intensity(spec: ModelSpec, prompt_tokens: int, dtype_bytes: int=2) -> float:
    """Prefill processes ``prompt_tokens`` tokens per weight read: intensity ≈ 2 N T / (N · bytes) = 2T/bytes."""
    raise NotImplementedError('TODO: implement prefill_arithmetic_intensity (see the reference in src/mlbook)')

def human_bytes(n: float) -> str:
    """Format bytes as KiB/MiB/GiB/TiB."""
    raise NotImplementedError('TODO: implement human_bytes (see the reference in src/mlbook)')
