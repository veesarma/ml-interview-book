"""Autoregressive decoding: greedy / temperature / top-k / top-p, with and without a KV cache.

The cached loop has two phases:
    prefill: logits = model(prompt, cache)         one forward over T_prompt tokens
    decode:  logits = model(next_token, cache)     one forward over ONE token per step
Both produce token-for-token identical output to recomputing the full prefix every step
(tested in tests/test_transformer_gpt.py).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .gpt import GPT


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token id from (B, V) logits. Returns (B,) long tensor.

    temperature: divide logits by tau (tau -> 0 approaches greedy, tau > 1 flattens).
    top_k: keep only the k largest logits.
    top_p: keep the smallest set of tokens whose cumulative probability >= p (nucleus).
    """
    if greedy:
        return logits.argmax(dim=-1)  # (B,)
    logits = logits / max(temperature, 1e-8)  # (B, V)
    if top_k is not None:
        kth = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]  # (B, 1) k-th largest
        logits = logits.masked_fill(logits < kth, float("-inf"))  # (B, V)
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)  # (B, V)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (B, V)
        remove = cum - F.softmax(sorted_logits, dim=-1) >= top_p  # (B, V) drop tokens after the nucleus is full
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)  # (B, V) back in vocab order
    probs = F.softmax(logits, dim=-1)  # (B, V)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)  # (B,)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend prompt ``idx`` (B, T_prompt) by ``max_new_tokens``. Returns (B, T_prompt + max_new_tokens)."""
    model.eval()
    B = idx.shape[0]
    if use_cache:
        cache = model.new_cache(B, idx.device)
        logits, _ = model(idx, cache=cache)  # prefill: (B, T_prompt, V); cache.length = T_prompt
        for _ in range(max_new_tokens):
            next_tok = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)  # (B,)
            idx = torch.cat([idx, next_tok[:, None]], dim=1)  # (B, T + 1)
            logits, _ = model(next_tok[:, None], cache=cache)  # decode: (B, 1, V); cache grows by one
        return idx
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size :]  # (B, <= T_max) crop to the context window
        logits, _ = model(idx_cond)  # (B, T, V) recompute everything
        next_tok = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)  # (B,)
        idx = torch.cat([idx, next_tok[:, None]], dim=1)  # (B, T + 1)
    return idx
