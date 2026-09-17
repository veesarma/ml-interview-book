# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/memory_calc.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k memory_calc -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/memory_calc --force

"""Training-memory accounting for dense Transformers (pure Python / NumPy).

    memory_per_gpu = weights + gradients + optimizer_states + activations (+ workspace)

Bytes per parameter for Adam in bf16 mixed precision (the number to reproduce at a
whiteboard):

    2 (bf16 weight) + 2 (bf16 grad) + 4 (fp32 master) + 4 (Adam m) + 4 (Adam v) = 16 B/param

ZeRO stages shard the states across the data-parallel group of size ``dp``:

    stage 1: optimizer states / dp
    stage 2: + gradients / dp
    stage 3: + weights / dp            (FSDP "full shard" == ZeRO-3)

Activation memory per layer (Korthikanti et al. 2022, "Reducing Activation
Recomputation in Large Transformer Models", 16-bit activations, classic GELU MLP
with d_ff = 4h, attention scores materialised):

    A_layer = s * b * h * (34 + 5 * a * s / h)      bytes, no parallelism
    A_layer = s * b * h * (10 + 24/t + 5 a s / (h t))   tensor parallel degree t
    A_layer = s * b * h * (34/t + 5 a s / (h t))       t + sequence parallel
    A_layer = s * b * h * 34 / t                        + selective recompute
    A_layer = 2 * s * b * h                             full recompute (layer input only)

where s = sequence length, b = micro-batch, h = hidden size, a = attention heads.
"""
from __future__ import annotations
from dataclasses import dataclass
BYTES_PER_DTYPE: dict[str, float] = {'fp32': 4.0, 'tf32': 4.0, 'fp16': 2.0, 'bf16': 2.0, 'fp8': 1.0, 'int8': 1.0, 'int4': 0.5}
GB = 1024 ** 3

@dataclass(frozen=True)
class TransformerConfig:
    """Shapes of a decoder-only Transformer.

    ``n_kv_heads`` < ``n_heads`` means grouped-query attention; ``gated_mlp`` means
    SwiGLU-style (three matrices of shape (h, d_ff)) instead of two.
    """
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    vocab: int
    n_kv_heads: int | None = None
    gated_mlp: bool = True
    tie_embeddings: bool = False

    @property
    def d_head(self) -> int:
        raise NotImplementedError('TODO: implement d_head (see the reference in src/mlbook)')

    @property
    def kv_heads(self) -> int:
        raise NotImplementedError('TODO: implement kv_heads (see the reference in src/mlbook)')
LLAMA2_7B = TransformerConfig(32, 4096, 32, 11008, 32000, n_kv_heads=32)
LLAMA3_8B = TransformerConfig(32, 4096, 32, 14336, 128256, n_kv_heads=8)
LLAMA3_70B = TransformerConfig(80, 8192, 64, 28672, 128256, n_kv_heads=8)
LLAMA3_405B = TransformerConfig(126, 16384, 128, 53248, 128256, n_kv_heads=8)

def count_params(cfg: TransformerConfig) -> dict[str, int]:
    """Parameter count by component (biases ignored, RMSNorm scales counted)."""
    raise NotImplementedError('TODO: implement count_params (see the reference in src/mlbook)')

def bytes_per_param_training(weight_dtype: str='bf16', grad_dtype: str='bf16', master_weights: bool=True, optimizer: str='adam') -> dict[str, float]:
    """Bytes of persistent state per parameter for training.

    ``optimizer`` in {"adam", "sgd_momentum", "sgd", "adafactor_like"} (adafactor is
    approximated as 0.5 B/param of factored moments, an order-of-magnitude figure).
    """
    raise NotImplementedError('TODO: implement bytes_per_param_training (see the reference in src/mlbook)')

def activation_bytes_per_layer(s: int, b: int, h: int, a: int, tp: int=1, sequence_parallel: bool=False, recompute: str='none', flash_attention: bool=False) -> float:
    """Korthikanti et al. activation bytes for one Transformer layer (16-bit).

    ``recompute`` in {"none", "selective", "full"}. ``flash_attention=True`` drops the
    materialised (a, s, s) score/softmax/dropout tensors, which is the same saving
    selective recomputation buys, without the recompute FLOPs.
    """
    raise NotImplementedError('TODO: implement activation_bytes_per_layer (see the reference in src/mlbook)')

@dataclass(frozen=True)
class ParallelPlan:
    """Degrees of parallelism. ``dp * tp * pp * cp`` must equal the GPU count."""
    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    zero_stage: int = 0

    @property
    def world_size(self) -> int:
        raise NotImplementedError('TODO: implement world_size (see the reference in src/mlbook)')

@dataclass(frozen=True)
class MemoryBreakdown:
    weights: float
    grads: float
    optimizer: float
    activations: float

    @property
    def total(self) -> float:
        raise NotImplementedError('TODO: implement total (see the reference in src/mlbook)')

    def as_gb(self) -> dict[str, float]:
        raise NotImplementedError('TODO: implement as_gb (see the reference in src/mlbook)')

def training_memory_per_gpu(cfg: TransformerConfig, plan: ParallelPlan, seq_len: int, micro_batch: int, recompute: str='none', sequence_parallel: bool=False, flash_attention: bool=True, weight_dtype: str='bf16', grad_dtype: str='bf16', optimizer: str='adam', pipeline_schedule: str='1f1b') -> MemoryBreakdown:
    """Per-GPU training memory (bytes) for ``cfg`` under ``plan``.

    Parameters are split ``tp * pp`` ways by model parallelism, then the persistent
    training states are further split ``dp`` ways according to the ZeRO stage.
    Activations: the first pipeline stage under 1F1B holds ``pp`` micro-batches in
    flight, so its activation footprint is ``n_layers`` layers' worth regardless of
    ``pp`` (GPipe holds all micro-batches; we report the 1F1B first-stage worst case).
    Context parallelism divides the per-layer activations by ``cp``.
    """
    raise NotImplementedError('TODO: implement training_memory_per_gpu (see the reference in src/mlbook)')

def zero_stage_table(cfg: TransformerConfig, dp: int, tp: int=1, pp: int=1) -> dict[int, dict[str, float]]:
    """State memory (GB) per GPU for ZeRO stages 0..3 with ``dp`` data-parallel ranks."""
    raise NotImplementedError('TODO: implement zero_stage_table (see the reference in src/mlbook)')

def kv_cache_bytes_per_token(cfg: TransformerConfig, dtype: str='bf16') -> float:
    """KV bytes per token: 2 (K and V) * L * n_kv_heads * d_head * bytes."""
    raise NotImplementedError('TODO: implement kv_cache_bytes_per_token (see the reference in src/mlbook)')
