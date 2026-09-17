# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/token_compression.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k token_compression -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/token_compression --force

"""Ways to reduce the number of visual tokens fed to an LLM.

All functions take a grid of visual tokens ``(B, h, w, d)`` or a sequence
``(B, N, d)`` and return fewer tokens. The KV cache and prefill cost of the
LLM scale with the number of tokens, so these are the levers that decide
whether a high-resolution or multi-image request fits a latency budget.
"""
from __future__ import annotations
import torch
from torch import nn

def average_pool_tokens(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Average each k x k block: (B, h, w, d) -> (B, (h/k)*(w/k), d)."""
    raise NotImplementedError('TODO: implement average_pool_tokens (see the reference in src/mlbook)')

def pixel_shuffle_merge(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Space-to-depth ("pixel unshuffle", InternVL / Qwen-VL style): (B, h, w, d) -> (B, (h/k)*(w/k), k*k*d).

    No information is discarded: k x k neighbouring tokens are CONCATENATED along the channel
    axis, and an MLP afterwards maps k*k*d -> d_llm. Token count drops by k^2.
    """
    raise NotImplementedError('TODO: implement pixel_shuffle_merge (see the reference in src/mlbook)')

class PixelShuffleProjector(nn.Module):
    """pixel_shuffle_merge followed by an MLP: (B, h, w, d_v) -> (B, N/k^2, d_llm)."""

    def __init__(self, d_v: int, d_llm: int, k: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def prune_tokens_by_score(tokens: torch.Tensor, scores: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``keep`` highest-scoring tokens per example (order preserved).

    ``tokens`` (B, N, d), ``scores`` (B, N) e.g. CLS-attention or text-relevance. Returns
    (kept (B, keep, d), indices (B, keep)).
    """
    raise NotImplementedError('TODO: implement prune_tokens_by_score (see the reference in src/mlbook)')

def anyres_tiles(image: torch.Tensor, tile: int) -> torch.Tensor:
    """Split a (B, C, H, W) image into non-overlapping tile x tile crops: (B * n_tiles, C, tile, tile).

    LLaVA-NeXT "AnyRes" style: a high-resolution image is encoded as several fixed-resolution
    tiles plus one global downscaled view (the caller adds the global view).
    """
    raise NotImplementedError('TODO: implement anyres_tiles (see the reference in src/mlbook)')

def visual_token_count(H: int, W: int, patch: int, merge: int=1, n_tiles: int=1, n_queries: int | None=None) -> int:
    """How many visual tokens an image contributes to the LLM sequence.

    Patch grid (H/P)(W/P) per tile, divided by merge^2 for pixel-shuffle/pooling, times tiles;
    a resampler overrides everything with its fixed n_queries.
    """
    raise NotImplementedError('TODO: implement visual_token_count (see the reference in src/mlbook)')
