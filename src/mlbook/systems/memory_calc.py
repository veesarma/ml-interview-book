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

BYTES_PER_DTYPE: dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,  # stored as fp32, computed with 10-bit mantissa
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}

GB = 1024**3


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
        return self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads or self.n_heads


# Public architectures, from the Llama 2 / Llama 3 papers and model cards.
LLAMA2_7B = TransformerConfig(32, 4096, 32, 11008, 32000, n_kv_heads=32)
LLAMA3_8B = TransformerConfig(32, 4096, 32, 14336, 128256, n_kv_heads=8)
LLAMA3_70B = TransformerConfig(80, 8192, 64, 28672, 128256, n_kv_heads=8)
LLAMA3_405B = TransformerConfig(126, 16384, 128, 53248, 128256, n_kv_heads=8)


def count_params(cfg: TransformerConfig) -> dict[str, int]:
    """Parameter count by component (biases ignored, RMSNorm scales counted)."""
    h, L = cfg.d_model, cfg.n_layers
    kv_dim = cfg.kv_heads * cfg.d_head
    attn = h * h + 2 * h * kv_dim + h * h  # W_q, W_k, W_v, W_o
    mlp = (3 if cfg.gated_mlp else 2) * h * cfg.d_ff
    norms = 2 * h
    per_layer = attn + mlp + norms
    embed = cfg.vocab * h
    unembed = 0 if cfg.tie_embeddings else cfg.vocab * h
    total = L * per_layer + embed + unembed + h  # + final norm
    return {
        "attention": L * attn,
        "mlp": L * mlp,
        "norms": L * norms + h,
        "embedding": embed,
        "unembedding": unembed,
        "total": total,
    }


def bytes_per_param_training(
    weight_dtype: str = "bf16",
    grad_dtype: str = "bf16",
    master_weights: bool = True,
    optimizer: str = "adam",
) -> dict[str, float]:
    """Bytes of persistent state per parameter for training.

    ``optimizer`` in {"adam", "sgd_momentum", "sgd", "adafactor_like"} (adafactor is
    approximated as 0.5 B/param of factored moments, an order-of-magnitude figure).
    """
    w = BYTES_PER_DTYPE[weight_dtype]
    g = BYTES_PER_DTYPE[grad_dtype]
    master = 4.0 if master_weights and weight_dtype != "fp32" else 0.0
    moments = {"adam": 8.0, "sgd_momentum": 4.0, "sgd": 0.0, "adafactor_like": 0.5}[optimizer]
    return {"weights": w, "grads": g, "master": master, "optimizer_moments": moments,
            "total": w + g + master + moments}


def activation_bytes_per_layer(
    s: int,
    b: int,
    h: int,
    a: int,
    tp: int = 1,
    sequence_parallel: bool = False,
    recompute: str = "none",
    flash_attention: bool = False,
) -> float:
    """Korthikanti et al. activation bytes for one Transformer layer (16-bit).

    ``recompute`` in {"none", "selective", "full"}. ``flash_attention=True`` drops the
    materialised (a, s, s) score/softmax/dropout tensors, which is the same saving
    selective recomputation buys, without the recompute FLOPs.
    """
    sbh = float(s * b * h)
    if recompute == "full":
        return 2.0 * sbh  # only the layer input is kept
    if recompute == "selective" or flash_attention:
        return sbh * 34.0 / tp if (sequence_parallel or tp == 1) else sbh * (10.0 + 24.0 / tp)
    score_term = 5.0 * a * s / (h * tp)
    if sequence_parallel or tp == 1:
        return sbh * (34.0 / tp + score_term)
    return sbh * (10.0 + 24.0 / tp + score_term)


@dataclass(frozen=True)
class ParallelPlan:
    """Degrees of parallelism. ``dp * tp * pp * cp`` must equal the GPU count."""

    dp: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    zero_stage: int = 0  # 0 = plain DDP, 1/2/3 = ZeRO-1/2/3 (3 == FSDP full shard)

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp * self.cp


@dataclass(frozen=True)
class MemoryBreakdown:
    weights: float
    grads: float
    optimizer: float
    activations: float

    @property
    def total(self) -> float:
        return self.weights + self.grads + self.optimizer + self.activations

    def as_gb(self) -> dict[str, float]:
        return {k: getattr(self, k) / GB for k in ("weights", "grads", "optimizer", "activations", "total")}


def training_memory_per_gpu(
    cfg: TransformerConfig,
    plan: ParallelPlan,
    seq_len: int,
    micro_batch: int,
    recompute: str = "none",
    sequence_parallel: bool = False,
    flash_attention: bool = True,
    weight_dtype: str = "bf16",
    grad_dtype: str = "bf16",
    optimizer: str = "adam",
    pipeline_schedule: str = "1f1b",
) -> MemoryBreakdown:
    """Per-GPU training memory (bytes) for ``cfg`` under ``plan``.

    Parameters are split ``tp * pp`` ways by model parallelism, then the persistent
    training states are further split ``dp`` ways according to the ZeRO stage.
    Activations: the first pipeline stage under 1F1B holds ``pp`` micro-batches in
    flight, so its activation footprint is ``n_layers`` layers' worth regardless of
    ``pp`` (GPipe holds all micro-batches; we report the 1F1B first-stage worst case).
    Context parallelism divides the per-layer activations by ``cp``.
    """
    params = count_params(cfg)["total"]
    params_per_gpu = params / (plan.tp * plan.pp)
    bpp = bytes_per_param_training(weight_dtype, grad_dtype, True, optimizer)
    shard_w = plan.dp if plan.zero_stage >= 3 else 1
    shard_g = plan.dp if plan.zero_stage >= 2 else 1
    shard_o = plan.dp if plan.zero_stage >= 1 else 1
    weights = params_per_gpu * bpp["weights"] / shard_w
    grads = params_per_gpu * bpp["grads"] / shard_g
    optim = params_per_gpu * (bpp["master"] + bpp["optimizer_moments"]) / shard_o

    per_layer = activation_bytes_per_layer(
        seq_len, micro_batch, cfg.d_model, cfg.n_heads, plan.tp, sequence_parallel,
        recompute, flash_attention,
    ) / plan.cp
    layers_per_stage = cfg.n_layers / plan.pp
    in_flight = plan.pp if pipeline_schedule == "1f1b" else plan.pp  # first-stage worst case
    activations = per_layer * layers_per_stage * in_flight
    return MemoryBreakdown(weights, grads, optim, activations)


def zero_stage_table(cfg: TransformerConfig, dp: int, tp: int = 1, pp: int = 1) -> dict[int, dict[str, float]]:
    """State memory (GB) per GPU for ZeRO stages 0..3 with ``dp`` data-parallel ranks."""
    out = {}
    for stage in range(4):
        plan = ParallelPlan(dp=dp, tp=tp, pp=pp, zero_stage=stage)
        m = training_memory_per_gpu(cfg, plan, seq_len=1, micro_batch=1, recompute="full")
        out[stage] = {"weights": m.weights / GB, "grads": m.grads / GB, "optimizer": m.optimizer / GB,
                      "states_total": (m.weights + m.grads + m.optimizer) / GB}
    return out


def kv_cache_bytes_per_token(cfg: TransformerConfig, dtype: str = "bf16") -> float:
    """KV bytes per token: 2 (K and V) * L * n_kv_heads * d_head * bytes."""
    return 2.0 * cfg.n_layers * cfg.kv_heads * cfg.d_head * BYTES_PER_DTYPE[dtype]
