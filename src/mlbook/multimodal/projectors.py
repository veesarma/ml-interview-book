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
        super().__init__()
        self.proj = nn.Linear(d_v, d_llm)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.proj(feats)  # (B, N_v, d_llm)


class MLPProjector(nn.Module):
    """(B, N_v, d_v) -> (B, N_v, d_llm) via Linear(d_v, d_llm) -> GELU -> Linear(d_llm, d_llm)."""

    def __init__(self, d_v: int, d_llm: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_v, d_llm)
        self.fc2 = nn.Linear(d_llm, d_llm)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(feats))  # (B, N_v, d_llm)
        return self.fc2(h)  # (B, N_v, d_llm)


class PerceiverResampler(nn.Module):
    """Learned queries attend to visual features: (B, N_v, d_v) -> (B, N_q, d_llm).

    Each layer: q = q + CrossAttn(LN(q), feats); q = q + MLP(LN(q)). Output length is
    fixed at N_q regardless of N_v (and of resolution), which bounds the LLM cost.
    """

    def __init__(self, d_v: int, d_llm: int, n_queries: int, n_heads: int, depth: int = 2, d_inner: int | None = None) -> None:
        super().__init__()
        d = d_inner or d_llm
        self.queries = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)  # (1, N_q, d)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "ln_q": nn.LayerNorm(d), "ln_kv": nn.LayerNorm(d_v),
                "xattn": MultiHeadCrossAttention(d, d_v, n_heads),
                "ln_mlp": nn.LayerNorm(d), "mlp": MLP(d),
            }))
        self.out = nn.Linear(d, d_llm)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, N_q, d)
        for layer in self.layers:
            q = q + layer["xattn"](layer["ln_q"](q), layer["ln_kv"](feats))  # (B, N_q, d)
            q = q + layer["mlp"](layer["ln_mlp"](q))  # (B, N_q, d)
        return self.out(q)  # (B, N_q, d_llm)


class QFormer(nn.Module):
    """BLIP-2 style Q-Former: learned queries with self-attention AND cross-attention per layer.

    (B, N_v, d_v) -> (B, N_q, d_llm). Differs from the Perceiver resampler by letting queries
    talk to each other (self-attention) before reading the image.
    """

    def __init__(self, d_v: int, d_llm: int, n_queries: int, n_heads: int, depth: int = 2, d_inner: int | None = None) -> None:
        super().__init__()
        d = d_inner or d_llm
        self.queries = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)  # (1, N_q, d)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                "ln_s": nn.LayerNorm(d), "self_attn": MultiHeadSelfAttention(d, n_heads),
                "ln_q": nn.LayerNorm(d), "ln_kv": nn.LayerNorm(d_v),
                "xattn": MultiHeadCrossAttention(d, d_v, n_heads),
                "ln_mlp": nn.LayerNorm(d), "mlp": MLP(d),
            }))
        self.out = nn.Linear(d, d_llm)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        q = self.queries.expand(B, -1, -1)  # (B, N_q, d)
        for layer in self.layers:
            q = q + layer["self_attn"](layer["ln_s"](q))  # (B, N_q, d)
            q = q + layer["xattn"](layer["ln_q"](q), layer["ln_kv"](feats))  # (B, N_q, d)
            q = q + layer["mlp"](layer["ln_mlp"](q))  # (B, N_q, d)
        return self.out(q)  # (B, N_q, d_llm)


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
        super().__init__()
        self.ln_t = nn.LayerNorm(d_llm)
        self.ln_v = nn.LayerNorm(d_v)
        self.xattn = MultiHeadCrossAttention(d_llm, d_v, n_heads)
        self.gate_attn = nn.Parameter(torch.zeros(1))  # scalar gate, tanh(0) = 0
        self.ln_m = nn.LayerNorm(d_llm)
        self.mlp = MLP(d_llm)
        self.gate_mlp = nn.Parameter(torch.zeros(1))  # scalar gate

    def forward(self, text: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        text = text + torch.tanh(self.gate_attn) * self.xattn(self.ln_t(text), self.ln_v(feats))  # (B, T, d_llm)
        text = text + torch.tanh(self.gate_mlp) * self.mlp(self.ln_m(text))  # (B, T, d_llm)
        return text
