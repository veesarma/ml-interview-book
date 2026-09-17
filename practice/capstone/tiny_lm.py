# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/tiny_lm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tiny_lm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/tiny_lm --force

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
MASK_VALUE = -1000000000.0
'Additive mask value.\n\nNot ``-inf``: a right-padded batch has query rows that attend to nothing, and\n``softmax`` of an all-``-inf`` row is ``NaN``, which then poisons the whole\nbackward pass. A large finite negative number gives a harmless uniform row\ninstead. Under fp16 anything below about ``-6.5e4`` overflows to ``-inf``, so\nproduction code uses ``torch.finfo(dtype).min`` rather than a hard-coded\nconstant.\n'

def causal_padding_bias(attention_mask: torch.Tensor) -> torch.Tensor:
    """Combine a causal mask with a key-padding mask into an additive bias.

    Args:
        attention_mask: ``(B, T)``, 1 for real tokens and 0 for padding.
    Returns:
        ``(B, 1, T, T)`` additive bias, 0 where attention is allowed and
        ``MASK_VALUE`` where it is not. The head axis is 1 so it broadcasts.
    """
    raise NotImplementedError('TODO: implement causal_padding_bias (see the reference in src/mlbook)')

class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention, three separate projections. ``(B, T, d) -> (B, T, d)``."""

    def __init__(self, d: int, n_heads: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class DecoderBlock(nn.Module):
    """Pre-norm decoder block. ``(B, T, d) -> (B, T, d)``."""

    def __init__(self, d: int, n_heads: int, mlp_ratio: int=4) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyCausalLM(nn.Module):
    """A small explicit GPT: learned token and position embeddings, ``n_layers`` blocks, tied-free head."""

    def __init__(self, vocab_size: int, d_model: int=48, n_heads: int=4, n_layers: int=2, max_positions: int=32) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``(B, T)`` ids -> ``(B, T, d)`` embeddings, *without* positions added."""
        raise NotImplementedError('TODO: implement embed_tokens (see the reference in src/mlbook)')

    def encode(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Run the stack on arbitrary embeddings.

        Args:
            inputs_embeds: ``(B, T, d)`` token embeddings, visual or textual.
            attention_mask: ``(B, T)`` 1 for real positions, 0 for padding.
            position_ids: ``(B, T)`` integer positions into ``pos_embed``.
        Returns:
            ``(B, T, d)`` final hidden states.
        """
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None, position_ids: torch.Tensor | None=None) -> torch.Tensor:
        """``(B, T)`` ids -> ``(B, T, V)`` logits. ``logits[:, t]`` scores token ``t + 1``."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, greedy: bool=True, temperature: float=1.0, top_k: int | None=None, eos_id: int | None=None, generator: torch.Generator | None=None) -> torch.Tensor:
        """Autoregressive decoding without a cache. ``(B, T)`` -> ``(B, T + max_new_tokens)``.

        ``greedy=True`` takes the argmax; otherwise it samples from
        ``softmax(logits / temperature)`` after an optional top-``k`` truncation.
        """
        raise NotImplementedError('TODO: implement generate (see the reference in src/mlbook)')

def sample_next_token(logits: torch.Tensor, greedy: bool=True, temperature: float=1.0, top_k: int | None=None, generator: torch.Generator | None=None) -> torch.Tensor:
    """Pick one token per row. ``(B, V)`` logits -> ``(B,)`` ids.

    Greedy is the ``temperature -> 0`` limit; sampling divides by the temperature
    and optionally keeps only the ``top_k`` logits, masking the rest.
    """
    raise NotImplementedError('TODO: implement sample_next_token (see the reference in src/mlbook)')
