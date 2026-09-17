# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/kv_cache.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k kv_cache -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/kv_cache --force

"""An explicit key/value cache for autoregressive decoding.

During generation, token t attends to keys/values of tokens 0..t. Without a cache,
every step recomputes K and V for the whole prefix (O(T^2) total projection work);
with it, each step projects only the *new* token and appends:

    prefill: run the prompt (B, T_prompt) once, store K, V per layer
             cache shape per layer: (B, H, T_prompt, d_head)
    decode:  run one token (B, 1), append its K, V -> (B, H, T_prompt + 1, d_head)
             and attend the single query against all cached keys.

Memory per token = 2 (K and V) * L * H * d_head * bytes = 2 * L * d_model * bytes,
which is why serving systems (vLLM's PagedAttention) treat the cache as the
scarce resource.
"""
from __future__ import annotations
import torch

class LayerKVCache:
    """Pre-allocated K/V buffers for one attention layer.

    Buffers: ``k``, ``v`` of shape (B, H, T_max, d_head); ``length`` tokens are valid.
    """

    def __init__(self, B: int, H: int, T_max: int, d_head: int, dtype: torch.dtype=torch.float32, device: torch.device | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def update(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new keys/values and return the full valid prefix.

        Args:
            k_new, v_new: (B, H, T_new, d_head); T_new = prompt length at prefill, 1 at decode.
        Returns:
            k_all, v_all: (B, H, length + T_new, d_head) views over the buffers.
        """
        raise NotImplementedError('TODO: implement update (see the reference in src/mlbook)')

    def reset(self) -> None:
        raise NotImplementedError('TODO: implement reset (see the reference in src/mlbook)')

class KVCache:
    """One ``LayerKVCache`` per Transformer layer."""

    def __init__(self, n_layers: int, B: int, H: int, T_max: int, d_head: int, dtype: torch.dtype=torch.float32, device: torch.device | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def __getitem__(self, i: int) -> LayerKVCache:
        raise NotImplementedError('TODO: implement __getitem__ (see the reference in src/mlbook)')

    def __len__(self) -> int:
        raise NotImplementedError('TODO: implement __len__ (see the reference in src/mlbook)')

    @property
    def length(self) -> int:
        """Number of cached positions (identical across layers)."""
        raise NotImplementedError('TODO: implement length (see the reference in src/mlbook)')

    def reset(self) -> None:
        raise NotImplementedError('TODO: implement reset (see the reference in src/mlbook)')

def kv_cache_bytes(n_layers: int, B: int, n_kv_heads: int, T: int, d_head: int, bytes_per_element: int=2) -> int:
    """Bytes held by the cache: 2 (K and V) * L * B * H_kv * T * d_head * bytes.

    With multi-query / grouped-query attention ``n_kv_heads`` < ``n_heads`` shrinks this
    linearly -- the main reason those variants exist.
    """
    raise NotImplementedError('TODO: implement kv_cache_bytes (see the reference in src/mlbook)')
