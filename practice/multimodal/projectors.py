# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/multimodal/projectors.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k projectors -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py multimodal/projectors --force

"""Vision -> LLM projectors. Every module maps visual features ``(B, N_v, d_v)``
to LLM-space tokens ``(B, N_q, d_llm)``:

* ``LinearProjector``     N_q = N_v      (LLaVA-1)
* ``MLPProjector``        N_q = N_v      (LLaVA-1.5: Linear-GELU-Linear)
* ``PerceiverResampler``  N_q = fixed    (Flamingo / BLIP-2 Q-Former style: learned queries cross-attend to features)
* ``GatedCrossAttentionAdapter``  the visual tokens are NOT inserted into the LLM sequence; instead
                          text hidden states (B, T, d_llm) cross-attend to them through a tanh-gated
                          block initialised at zero (Flamingo, Llama 3.2-Vision).
"""
from __future__ import annotations
import torch
from torch import nn
from .attention_block import MLP, MultiHeadCrossAttention, MultiHeadSelfAttention

class LinearProjector(nn.Module):
    """(B, N_v, d_v) -> (B, N_v, d_llm). One matrix W in R^{d_v x d_llm}."""

    def __init__(self, d_v: int, d_llm: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class MLPProjector(nn.Module):
    """(B, N_v, d_v) -> (B, N_v, d_llm) via Linear(d_v, d_llm) -> GELU -> Linear(d_llm, d_llm)."""

    def __init__(self, d_v: int, d_llm: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class PerceiverResampler(nn.Module):
    """Learned queries attend to visual features: (B, N_v, d_v) -> (B, N_q, d_llm).

    Each layer: q = q + CrossAttn(LN(q), feats); q = q + MLP(LN(q)). Output length is
    fixed at N_q regardless of N_v (and of resolution), which bounds the LLM cost.
    """

    def __init__(self, d_v: int, d_llm: int, n_queries: int, n_heads: int, depth: int=2, d_inner: int | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class QFormer(nn.Module):
    """BLIP-2 style Q-Former: learned queries with self-attention AND cross-attention per layer.

    (B, N_v, d_v) -> (B, N_q, d_llm). Differs from the Perceiver resampler by letting queries
    talk to each other (self-attention) before reading the image.
    """

    def __init__(self, d_v: int, d_llm: int, n_queries: int, n_heads: int, depth: int=2, d_inner: int | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class GatedCrossAttentionAdapter(nn.Module):
    """Flamingo-style gated cross-attention inserted between frozen LLM layers.

    ``text`` (B, T, d_llm) cross-attends to ``feats`` (B, N_v, d_v):
        text = text + tanh(g_a) * CrossAttn(LN(text), feats)
        text = text + tanh(g_m) * MLP(LN(text))
    with ``g_a = g_m = 0`` at init, so the LLM's behaviour is unchanged at step 0.
    Output (B, T, d_llm). The image never enters the token sequence, so the KV cache
    of the LLM does not grow with N_v.
    """

    def __init__(self, d_llm: int, d_v: int, n_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, text: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
