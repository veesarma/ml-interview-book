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
    B, h, w, d = grid.shape
    x = grid.view(B, h // k, k, w // k, k, d)  # (B, h/k, k, w/k, k, d)
    x = x.mean(dim=(2, 4))  # (B, h/k, w/k, d)
    return x.reshape(B, (h // k) * (w // k), d)  # (B, N/k^2, d)


def pixel_shuffle_merge(grid: torch.Tensor, k: int) -> torch.Tensor:
    """Space-to-depth ("pixel unshuffle", InternVL / Qwen-VL style): (B, h, w, d) -> (B, (h/k)*(w/k), k*k*d).

    No information is discarded: k x k neighbouring tokens are CONCATENATED along the channel
    axis, and an MLP afterwards maps k*k*d -> d_llm. Token count drops by k^2.
    """
    B, h, w, d = grid.shape
    x = grid.view(B, h // k, k, w // k, k, d)  # (B, h/k, k, w/k, k, d)
    x = x.permute(0, 1, 3, 2, 4, 5)  # (B, h/k, w/k, k, k, d)
    return x.reshape(B, (h // k) * (w // k), k * k * d)  # (B, N/k^2, k*k*d)


class PixelShuffleProjector(nn.Module):
    """pixel_shuffle_merge followed by an MLP: (B, h, w, d_v) -> (B, N/k^2, d_llm)."""

    def __init__(self, d_v: int, d_llm: int, k: int) -> None:
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(k * k * d_v)
        self.fc1 = nn.Linear(k * k * d_v, d_llm)
        self.fc2 = nn.Linear(d_llm, d_llm)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        x = self.norm(pixel_shuffle_merge(grid, self.k))  # (B, N/k^2, k*k*d_v)
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))  # (B, N/k^2, d_llm)


def prune_tokens_by_score(tokens: torch.Tensor, scores: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the ``keep`` highest-scoring tokens per example (order preserved).

    ``tokens`` (B, N, d), ``scores`` (B, N) e.g. CLS-attention or text-relevance. Returns
    (kept (B, keep, d), indices (B, keep)).
    """
    idx = scores.topk(keep, dim=1).indices  # (B, keep)
    idx = idx.sort(dim=1).values  # (B, keep), restore spatial order
    kept = torch.gather(tokens, 1, idx[:, :, None].expand(-1, -1, tokens.shape[-1]))  # (B, keep, d)
    return kept, idx


def anyres_tiles(image: torch.Tensor, tile: int) -> torch.Tensor:
    """Split a (B, C, H, W) image into non-overlapping tile x tile crops: (B * n_tiles, C, tile, tile).

    LLaVA-NeXT "AnyRes" style: a high-resolution image is encoded as several fixed-resolution
    tiles plus one global downscaled view (the caller adds the global view).
    """
    B, C, H, W = image.shape
    if H % tile or W % tile:
        raise ValueError("H and W must be multiples of the tile size")
    x = image.view(B, C, H // tile, tile, W // tile, tile)  # (B, C, nh, t, nw, t)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nh, nw, C, t, t)
    return x.reshape(B * (H // tile) * (W // tile), C, tile, tile)  # (B*n_tiles, C, t, t)


def visual_token_count(H: int, W: int, patch: int, merge: int = 1, n_tiles: int = 1, n_queries: int | None = None) -> int:
    """How many visual tokens an image contributes to the LLM sequence.

    Patch grid (H/P)(W/P) per tile, divided by merge^2 for pixel-shuffle/pooling, times tiles;
    a resampler overrides everything with its fixed n_queries.
    """
    if n_queries is not None:
        return n_queries
    return n_tiles * ((H // patch) * (W // patch)) // (merge * merge)
