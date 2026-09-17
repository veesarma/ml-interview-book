# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/sequence/seq2seq_attention.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k seq2seq_attention -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py sequence/seq2seq_attention --force

"""Encoder-decoder with Bahdanau (additive) or Luong (dot-product) attention.

The attention layer is a *soft dictionary lookup*: the decoder state is the query,
encoder states are the keys and (here) also the values,

    e_{t,j} = score(s_t, h_j)            one scalar per source position j
    alpha_t = softmax_j(e_t)             (T_src,) weights, sum to 1
    c_t     = sum_j alpha_{t,j} h_j      (d_enc,) context vector

with two scoring functions:

    additive (Bahdanau 2015):  score(s, h) = v^T tanh(s W_q + h W_k)
    dot / general (Luong 2015): score(s, h) = s W h^T   (or s h^T when d_dec == d_enc)

The recurrent units themselves are ``torch.nn.GRU`` (their maths is derived in
``mlbook.sequence.gru``); this module is about the attention bridge.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class AdditiveAttention(nn.Module):
    """Bahdanau attention. Query (B, d_q), keys (B, T_src, d_k) -> context (B, d_k), weights (B, T_src)."""

    def __init__(self, d_q: int, d_k: int, d_att: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class DotProductAttention(nn.Module):
    """Luong 'general' attention: score = s W h^T. Same signature as ``AdditiveAttention``."""

    def __init__(self, d_q: int, d_k: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class Seq2SeqAttention(nn.Module):
    """Bidirectional-GRU encoder + GRU decoder with attention over encoder states.

    forward(src, tgt_in) uses teacher forcing: the decoder input at step t is the
    *gold* token t-1, and the model predicts token t. Output logits: (B, T_tgt, V).
    """

    def __init__(self, vocab_size: int, d_model: int=32, attention: str='additive', pad_id: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """src (B, T_src) -> enc (B, T_src, 2d), s0 (B, d), src_mask (B, T_src) True where not pad."""
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode_step(self, y_prev: torch.Tensor, s: torch.Tensor, enc: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoder step. y_prev (B,), s (B, d) -> logits (B, V), new state (B, d), alpha (B, T_src)."""
        raise NotImplementedError('TODO: implement decode_step (see the reference in src/mlbook)')

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced training pass. src (B, T_src), tgt_in (B, T_tgt) -> logits (B, T_tgt, V), attn (B, T_tgt, T_src)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_id: int, max_len: int) -> torch.Tensor:
        """Free-running decoding: feed back the argmax. Returns (B, max_len) token ids."""
        raise NotImplementedError('TODO: implement greedy_decode (see the reference in src/mlbook)')

@torch.no_grad()
def beam_search(model: Seq2SeqAttention, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int, beam: int=3, length_alpha: float=0.6) -> list[int]:
    """Beam search for ONE source sequence (src: (1, T_src)).

    Keeps the ``beam`` highest log-prob prefixes; finished hypotheses are scored with
    GNMT-style length normalisation  lp = ((5 + L) / 6) ** alpha.
    Returns the best token list (without BOS, up to and excluding EOS).
    """
    raise NotImplementedError('TODO: implement beam_search (see the reference in src/mlbook)')
