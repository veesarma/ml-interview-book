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
    block_size: int  # maximum context length T_max
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int | None = None  # default 4 * d_model
    dropout: float = 0.0
    tie_weights: bool = True
    positional: str = "learned"  # "learned" or "rope"
    norm: str = "layer"  # "layer" or "rms"
    ffn: str = "gelu"  # "gelu" or "swiglu"


class GPT(nn.Module):
    """forward(idx (B, T), targets (B, T) | None, cache) -> (logits (B, T, V), loss | None)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # E: (V, d)
        self.pos_emb = LearnedPositionalEmbedding(cfg.block_size, cfg.d_model) if cfg.positional == "learned" else None
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads) if cfg.positional == "rope" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.norm, pre_norm=True, ffn=cfg.ffn, cross_attention=False)
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model) if cfg.norm == "layer" else RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # (d, V)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight  # share the (V, d) matrix
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, cache: KVCache | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        offset = 0 if cache is None else cache.length  # absolute position of idx[:, 0]
        if offset + T > self.cfg.block_size:
            raise ValueError(f"sequence of length {offset + T} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(idx)  # (B, T, d)
        if self.pos_emb is not None:
            x = x + self.pos_emb(T, offset)  # (B, T, d) + (1, T, d)
        x = self.drop(x)
        rope = self.rope(T, offset) if self.rope is not None else None  # (cos, sin) each (T, d_head)
        if cache is None:
            mask = causal_mask(T, idx.device)  # (1, 1, T, T)
        else:
            mask = causal_mask_with_cache(T, offset + T, idx.device)  # (1, 1, T, offset + T)
        for i, block in enumerate(self.blocks):
            x = block(x, None, mask, None, None if cache is None else cache[i], rope)  # (B, T, d)
        x = self.ln_f(x)  # (B, T, d)
        logits = self.lm_head(x)  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T), ignore_index=-100)
        return logits, loss

    def new_cache(self, B: int, device: torch.device | None = None) -> KVCache:
        return KVCache(self.cfg.n_layers, B, self.cfg.n_heads, self.cfg.block_size, self.cfg.d_model // self.cfg.n_heads, device=device)


def count_parameters(model: nn.Module) -> int:
    """Number of *unique* trainable parameters (tied weights counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def gpt_param_count(cfg: GPTConfig) -> int:
    """Closed-form parameter count for ``GPT`` (learned positions, LayerNorm, GELU FFN).

    Per layer: attention 4 d^2 + 4 d, FFN 2 d d_ff + d_ff + d, two LayerNorms 4 d.
    With d_ff = 4 d this is 12 d^2 + 13 d per layer -> the "12 L d^2" rule.
    Plus embeddings V d + T_max d, final LayerNorm 2 d, and V d more if the head is untied.
    """
    d, L, V = cfg.d_model, cfg.n_layers, cfg.vocab_size
    d_ff = 4 * d if cfg.d_ff is None else cfg.d_ff
    per_layer = (4 * d * d + 4 * d) + (2 * d * d_ff + d_ff + d) + 4 * d
    total = L * per_layer + V * d + 2 * d
    if cfg.positional == "learned":
        total += cfg.block_size * d
    if not cfg.tie_weights:
        total += V * d
    return total


def training_flops_per_token(n_params: int) -> int:
    """~6N: 2N for the forward matmuls, 4N for the backward (Kaplan et al., 2020), ignoring attention's T^2 term."""
    return 6 * n_params


def inference_flops_per_token(n_params: int) -> int:
    """~2N per generated token with a KV cache (one multiply-add per weight)."""
    return 2 * n_params
