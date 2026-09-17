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
    n_params: float  # total parameters (for weight-memory and FLOP estimates)

    @property
    def d_model(self) -> int:
        return self.n_heads * self.d_head


# Architecture numbers from the Llama 2 paper, the Llama 3 herd paper and the
# Mistral 7B paper (layers / heads / KV heads / head dim).
LLAMA2_7B = ModelSpec("Llama-2-7B", n_layers=32, n_heads=32, n_kv_heads=32, d_head=128, n_params=6.7e9)
LLAMA2_70B = ModelSpec("Llama-2-70B", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128, n_params=69e9)
LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, n_heads=32, n_kv_heads=8, d_head=128, n_params=8.0e9)
LLAMA3_70B = ModelSpec("Llama-3-70B", n_layers=80, n_heads=64, n_kv_heads=8, d_head=128, n_params=70e9)
LLAMA3_405B = ModelSpec("Llama-3-405B", n_layers=126, n_heads=128, n_kv_heads=8, d_head=128, n_params=405e9)
MISTRAL_7B = ModelSpec("Mistral-7B", n_layers=32, n_heads=32, n_kv_heads=8, d_head=128, n_params=7.2e9)
# A hypothetical MHA version of Llama-3-70B, to show what GQA saves.
LLAMA3_70B_MHA = ModelSpec("Llama-3-70B (if MHA)", n_layers=80, n_heads=64, n_kv_heads=64, d_head=128, n_params=70e9)


def kv_bytes_per_token(spec: ModelSpec, dtype_bytes: int = 2) -> int:
    """2 · L · H_kv · d_head · bytes."""
    return 2 * spec.n_layers * spec.n_kv_heads * spec.d_head * dtype_bytes


def kv_cache_bytes(spec: ModelSpec, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
    """Total KV-cache bytes for ``batch`` sequences of ``seq_len`` tokens."""
    return batch * seq_len * kv_bytes_per_token(spec, dtype_bytes)


def mla_bytes_per_token(n_layers: int, d_c: int = 512, d_rope: int = 64, dtype_bytes: int = 2) -> int:
    """MLA cache: L · (d_c + d_h^R) · bytes (DeepSeek-V2 defaults d_c=512, d_h^R=64)."""
    return n_layers * (d_c + d_rope) * dtype_bytes


def max_batch_at_context(spec: ModelSpec, seq_len: int, hbm_bytes: float, dtype_bytes: int = 2) -> int:
    """How many ``seq_len`` sequences fit after the weights are resident in ``hbm_bytes``."""
    free = hbm_bytes - spec.n_params * dtype_bytes
    if free <= 0:
        return 0
    return int(free // kv_cache_bytes(spec, seq_len, 1, dtype_bytes))


def decode_arithmetic_intensity(spec: ModelSpec, batch: int, context: int, dtype_bytes: int = 2) -> float:
    """FLOPs per HBM byte for one decode step of a batch (weights read once, KV read per sequence).

    FLOPs  ≈ 2 · N · batch        (one token per sequence through all weights)
    bytes  ≈ N · dtype + batch · context · kv_bytes_per_token
    """
    flops = 2.0 * spec.n_params * batch
    bytes_moved = spec.n_params * dtype_bytes + batch * context * kv_bytes_per_token(spec, dtype_bytes)
    return flops / bytes_moved


def prefill_arithmetic_intensity(spec: ModelSpec, prompt_tokens: int, dtype_bytes: int = 2) -> float:
    """Prefill processes ``prompt_tokens`` tokens per weight read: intensity ≈ 2 N T / (N · bytes) = 2T/bytes."""
    flops = 2.0 * spec.n_params * prompt_tokens
    bytes_moved = spec.n_params * dtype_bytes
    return flops / bytes_moved


def human_bytes(n: float) -> str:
    """Format bytes as KiB/MiB/GiB/TiB."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"
