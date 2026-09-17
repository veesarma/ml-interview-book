# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/gpt.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k gpt -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/gpt --force

"""A small but complete decoder-only Transformer (GPT-2 style) with an explicit KV cache.

    p(x_1..T) = prod_t p(x_t | x_<t)
    logits = LMHead(Block_L(... Block_1(TokEmb[x] + PosEmb[0..T-1])))
    loss   = mean_t CE(logits[t], x_{t+1})              (teacher forcing: inputs are gold prefixes)

Weight tying: the LM head reuses the token-embedding matrix E (V, d): logits = h E^T.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .blocks import RMSNorm, TransformerDecoderBlock
from .kv_cache import KVCache
from .masks import causal_mask, causal_mask_with_cache
from .positional import LearnedPositionalEmbedding, RotaryEmbedding

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int | None = None
    dropout: float = 0.0
    tie_weights: bool = True
    positional: str = 'learned'
    norm: str = 'layer'
    ffn: str = 'gelu'

class GPT(nn.Module):
    """forward(idx (B, T), targets (B, T) | None, cache) -> (logits (B, T, V), loss | None)."""

    def __init__(self, cfg: GPTConfig) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        raise NotImplementedError('TODO: implement _init_weights (see the reference in src/mlbook)')

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None=None, cache: KVCache | None=None) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def new_cache(self, B: int, device: torch.device | None=None) -> KVCache:
        raise NotImplementedError('TODO: implement new_cache (see the reference in src/mlbook)')

def count_parameters(model: nn.Module) -> int:
    """Number of *unique* trainable parameters (tied weights counted once)."""
    raise NotImplementedError('TODO: implement count_parameters (see the reference in src/mlbook)')

def gpt_param_count(cfg: GPTConfig) -> int:
    """Closed-form parameter count for ``GPT`` (learned positions, LayerNorm, GELU FFN).

    Per layer: attention 4 d^2 + 4 d, FFN 2 d d_ff + d_ff + d, two LayerNorms 4 d.
    With d_ff = 4 d this is 12 d^2 + 13 d per layer -> the "12 L d^2" rule.
    Plus embeddings V d + T_max d, final LayerNorm 2 d, and V d more if the head is untied.
    """
    raise NotImplementedError('TODO: implement gpt_param_count (see the reference in src/mlbook)')

def training_flops_per_token(n_params: int) -> int:
    """~6N: 2N for the forward matmuls, 4N for the backward (Kaplan et al., 2020), ignoring attention's T^2 term."""
    raise NotImplementedError('TODO: implement training_flops_per_token (see the reference in src/mlbook)')

def inference_flops_per_token(n_params: int) -> int:
    """~2N per generated token with a KV cache (one multiply-add per weight)."""
    raise NotImplementedError('TODO: implement inference_flops_per_token (see the reference in src/mlbook)')
