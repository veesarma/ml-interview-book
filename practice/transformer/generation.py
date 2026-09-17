# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/generation.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k generation -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/generation --force

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

def sample_next_token(logits: torch.Tensor, temperature: float=1.0, top_k: int | None=None, top_p: float | None=None, greedy: bool=False, generator: torch.Generator | None=None) -> torch.Tensor:
    """Pick the next token id from (B, V) logits. Returns (B,) long tensor.

    temperature: divide logits by tau (tau -> 0 approaches greedy, tau > 1 flattens).
    top_k: keep only the k largest logits.
    top_p: keep the smallest set of tokens whose cumulative probability >= p (nucleus).
    """
    raise NotImplementedError('TODO: implement sample_next_token (see the reference in src/mlbook)')

@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float=1.0, top_k: int | None=None, top_p: float | None=None, greedy: bool=False, use_cache: bool=True, generator: torch.Generator | None=None) -> torch.Tensor:
    """Extend prompt ``idx`` (B, T_prompt) by ``max_new_tokens``. Returns (B, T_prompt + max_new_tokens)."""
    raise NotImplementedError('TODO: implement generate (see the reference in src/mlbook)')
