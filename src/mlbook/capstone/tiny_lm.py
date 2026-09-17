"""Canon #60, part 3 -- the language model: an explicit causal Transformer LM.

This is canon #31 (tiny GPT) plus canon #32 (generation) written so the
multimodal wrapper can feed it *embeddings* instead of token ids. Everything a
VLM needs from its LM is here and nothing else:

* ``embed_tokens``   ids -> embeddings, so the caller can splice visual tokens in
* ``encode``         embeddings + mask + position ids -> final hidden states
* ``forward``        the usual ids -> logits path
* ``generate``       greedy or temperature/top-k sampling

No KV cache: that is canon #33 and it would double the length of this file for
sequences of sixteen tokens. The chapter says where it would go.

Shapes: ``B`` batch, ``T`` sequence length, ``d`` model width, ``H`` heads,
``d_head = d / H``, ``V`` vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MASK_VALUE = -1e9
"""Additive mask value.

Not ``-inf``: a right-padded batch has query rows that attend to nothing, and
``softmax`` of an all-``-inf`` row is ``NaN``, which then poisons the whole
backward pass. A large finite negative number gives a harmless uniform row
instead. Under fp16 anything below about ``-6.5e4`` overflows to ``-inf``, so
production code uses ``torch.finfo(dtype).min`` rather than a hard-coded
constant.
"""


def causal_padding_bias(attention_mask: torch.Tensor) -> torch.Tensor:
    """Combine a causal mask with a key-padding mask into an additive bias.

    Args:
        attention_mask: ``(B, T)``, 1 for real tokens and 0 for padding.
    Returns:
        ``(B, 1, T, T)`` additive bias, 0 where attention is allowed and
        ``MASK_VALUE`` where it is not. The head axis is 1 so it broadcasts.
    """
    B, T = attention_mask.shape                                          # (B, T)
    device = attention_mask.device
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))  # (T, T)
    key_ok = attention_mask.bool()[:, None, None, :]                     # (B, 1, 1, T)
    allowed = causal[None, None, :, :] & key_ok                          # (B, 1, T, T)
    return torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, MASK_VALUE)


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention, three separate projections. ``(B, T, d) -> (B, T, d)``."""

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"width {d} is not divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape                                                # (B, T, d)
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        scores = q @ k.transpose(-2, -1) / self.d_head**0.5              # (B, H, T, T)
        scores = scores + bias                                           # (B, H, T, T) broadcast over H
        weights = torch.softmax(scores, dim=-1)                          # (B, H, T, T)
        context = weights @ v                                            # (B, H, T, d_head)
        context = context.transpose(1, 2).reshape(B, T, d)               # (B, T, d)
        return self.out_proj(context)                                    # (B, T, d)


class DecoderBlock(nn.Module):
    """Pre-norm decoder block. ``(B, T, d) -> (B, T, d)``."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), bias)                             # (B, T, d)
        x = x + self.mlp(self.ln2(x))                                    # (B, T, d)
        return x                                                         # (B, T, d)


class TinyCausalLM(nn.Module):
    """A small explicit GPT: learned token and position embeddings, ``n_layers`` blocks, tied-free head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        max_positions: int = 32,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_positions = max_positions
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_positions, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    # -- the three entry points ------------------------------------------------

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` ids -> ``(B, T, d)`` embeddings, *without* positions added."""
        return self.token_embed(input_ids)                               # (B, T, d)

    def encode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the stack on arbitrary embeddings.

        Args:
            inputs_embeds: ``(B, T, d)`` token embeddings, visual or textual.
            attention_mask: ``(B, T)`` 1 for real positions, 0 for padding.
            position_ids: ``(B, T)`` integer positions into ``pos_embed``.
        Returns:
            ``(B, T, d)`` final hidden states.
        """
        x = inputs_embeds + self.pos_embed(position_ids)                 # (B, T, d)
        bias = causal_padding_bias(attention_mask)                       # (B, 1, T, T)
        for block in self.blocks:
            x = block(x, bias)                                           # (B, T, d)
        return self.ln_f(x)                                              # (B, T, d)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, T)`` ids -> ``(B, T, V)`` logits. ``logits[:, t]`` scores token ``t + 1``."""
        B, T = input_ids.shape                                           # (B, T)
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=input_ids.device)  # (B, T)
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).expand(B, T)          # (B, T)
        h = self.encode(self.embed_tokens(input_ids), attention_mask, position_ids)        # (B, T, d)
        return self.lm_head(h)                                           # (B, T, V)

    # -- generation ------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoding without a cache. ``(B, T)`` -> ``(B, T + max_new_tokens)``.

        ``greedy=True`` takes the argmax; otherwise it samples from
        ``softmax(logits / temperature)`` after an optional top-``k`` truncation.
        """
        ids = input_ids                                                  # (B, T)
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)  # (B,)
        for _ in range(max_new_tokens):
            logits = self.forward(ids)[:, -1, :]                         # (B, V)
            next_id = sample_next_token(logits, greedy, temperature, top_k, generator)  # (B,)
            if eos_id is not None:
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)  # (B,)
                finished = finished | (next_id == eos_id)                # (B,)
            ids = torch.cat([ids, next_id[:, None]], dim=1)              # (B, T + 1)
        return ids                                                       # (B, T + max_new_tokens)


def sample_next_token(
    logits: torch.Tensor,
    greedy: bool = True,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row. ``(B, V)`` logits -> ``(B,)`` ids.

    Greedy is the ``temperature -> 0`` limit; sampling divides by the temperature
    and optionally keeps only the ``top_k`` logits, masking the rest.
    """
    if greedy:
        return logits.argmax(dim=-1)                                     # (B,)
    scaled = logits / max(temperature, 1e-6)                             # (B, V)
    if top_k is not None:
        kth = scaled.topk(top_k, dim=-1).values[:, -1:]                  # (B, 1)
        scaled = scaled.masked_fill(scaled < kth, MASK_VALUE)            # (B, V)
    probs = torch.softmax(scaled, dim=-1)                                # (B, V)
    return torch.multinomial(probs, num_samples=1, generator=generator)[:, 0]  # (B,)
