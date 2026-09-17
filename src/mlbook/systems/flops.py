"""FLOPs accounting for a dense decoder-only Transformer.

Per token, forward pass:

    F_fwd = 2 * P_matmul + 4 * L * h * s          (+ 2 * h * V for the LM head, in P_matmul)

* ``2 * P``: every weight element is used once in a multiply-accumulate (2 FLOPs).
* ``4 L h s``: attention scores Q K^T (2 s h per token per layer) and P V (2 s h).
  Causal masking could halve it; the convention (PaLM, Megatron) counts the full term.

Training = forward + backward = 3x forward, so training FLOPs/token = 6 P + 12 L h s.
With full activation recomputation the forward is done twice: 8 P + 16 L h s (this is
the difference between "model FLOPs" (MFU) and "hardware FLOPs" (HFU)).
"""

from __future__ import annotations

from mlbook.systems.memory_calc import TransformerConfig, count_params


def matmul_params(cfg: TransformerConfig) -> int:
    """Parameters that participate in matmuls (everything except norms and the
    input embedding lookup, which is a gather, not a matmul)."""
    p = count_params(cfg)
    unembed = cfg.vocab * cfg.d_model  # the LM head is a matmul even if tied
    return p["attention"] + p["mlp"] + unembed


def attention_flops_per_token_forward(cfg: TransformerConfig, seq_len: int) -> float:
    """4 * L * h * s: QK^T and PV, forward only, full (non-causal) count."""
    return 4.0 * cfg.n_layers * cfg.d_model * seq_len


def flops_per_token(cfg: TransformerConfig, seq_len: int, mode: str = "train", recompute: bool = False) -> float:
    """FLOPs per token for ``mode`` in {"forward", "train"}.

    train = 3 * forward; with ``recompute`` (full activation checkpointing) = 4 * forward.
    """
    fwd = 2.0 * matmul_params(cfg) + attention_flops_per_token_forward(cfg, seq_len)
    if mode == "forward":
        return fwd
    return (4.0 if recompute else 3.0) * fwd


def training_flops(cfg: TransformerConfig, tokens: float, seq_len: int) -> float:
    """Total training FLOPs ~ (6P + 12 L h s) * tokens."""
    return flops_per_token(cfg, seq_len, "train") * tokens


def six_nd(n_params: float, tokens: float) -> float:
    """The back-of-envelope C = 6 N D (ignores the attention term)."""
    return 6.0 * n_params * tokens
