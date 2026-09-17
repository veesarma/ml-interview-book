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
        super().__init__()
        self.W_q = nn.Linear(d_q, d_att, bias=False)  # (d_q, d_att)
        self.W_k = nn.Linear(d_k, d_att, bias=False)  # (d_k, d_att)
        self.v = nn.Linear(d_att, 1, bias=False)  # (d_att, 1)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W_q(query).unsqueeze(1)  # (B, 1, d_att)
        k = self.W_k(keys)  # (B, T_src, d_att)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # (B, T_src, 1) -> (B, T_src)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # (B, T_src)
        alpha = F.softmax(scores, dim=-1)  # (B, T_src)
        context = torch.bmm(alpha.unsqueeze(1), keys).squeeze(1)  # (B, 1, T_src) @ (B, T_src, d_k) -> (B, d_k)
        return context, alpha


class DotProductAttention(nn.Module):
    """Luong 'general' attention: score = s W h^T. Same signature as ``AdditiveAttention``."""

    def __init__(self, d_q: int, d_k: int) -> None:
        super().__init__()
        self.W = nn.Linear(d_q, d_k, bias=False)  # maps the query into key space

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W(query).unsqueeze(-1)  # (B, d_k, 1)
        scores = torch.bmm(keys, q).squeeze(-1)  # (B, T_src, d_k) @ (B, d_k, 1) -> (B, T_src)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)  # (B, T_src)
        alpha = F.softmax(scores, dim=-1)  # (B, T_src)
        context = torch.bmm(alpha.unsqueeze(1), keys).squeeze(1)  # (B, d_k)
        return context, alpha


class Seq2SeqAttention(nn.Module):
    """Bidirectional-GRU encoder + GRU decoder with attention over encoder states.

    forward(src, tgt_in) uses teacher forcing: the decoder input at step t is the
    *gold* token t-1, and the model predicts token t. Output logits: (B, T_tgt, V).
    """

    def __init__(self, vocab_size: int, d_model: int = 32, attention: str = "additive", pad_id: int = 0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)  # (V, d)
        self.encoder = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True)  # out (B, T, 2d)
        self.bridge = nn.Linear(2 * d_model, d_model)  # final encoder state -> decoder init
        if attention == "additive":
            self.attention: nn.Module = AdditiveAttention(d_q=d_model, d_k=2 * d_model, d_att=d_model)
        elif attention == "dot":
            self.attention = DotProductAttention(d_q=d_model, d_k=2 * d_model)
        else:
            raise ValueError(attention)
        self.decoder_cell = nn.GRUCell(d_model + 2 * d_model, d_model)  # input = [emb; context]
        self.out = nn.Linear(d_model + 2 * d_model, vocab_size)  # from [s_t; c_t]

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """src (B, T_src) -> enc (B, T_src, 2d), s0 (B, d), src_mask (B, T_src) True where not pad."""
        src_mask = src != self.pad_id  # (B, T_src)
        enc, h_n = self.encoder(self.embed(src))  # enc (B, T_src, 2d), h_n (2, B, d)
        s0 = torch.tanh(self.bridge(torch.cat([h_n[0], h_n[1]], dim=-1)))  # (B, d)
        return enc, s0, src_mask

    def decode_step(self, y_prev: torch.Tensor, s: torch.Tensor, enc: torch.Tensor, src_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoder step. y_prev (B,), s (B, d) -> logits (B, V), new state (B, d), alpha (B, T_src)."""
        context, alpha = self.attention(s, enc, src_mask)  # (B, 2d), (B, T_src)
        x = torch.cat([self.embed(y_prev), context], dim=-1)  # (B, 3d)
        s_new = self.decoder_cell(x, s)  # (B, d)
        logits = self.out(torch.cat([s_new, context], dim=-1))  # (B, V)
        return logits, s_new, alpha

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-forced training pass. src (B, T_src), tgt_in (B, T_tgt) -> logits (B, T_tgt, V), attn (B, T_tgt, T_src)."""
        enc, s, src_mask = self.encode(src)
        logits, attns = [], []
        for t in range(tgt_in.shape[1]):
            logit_t, s, alpha = self.decode_step(tgt_in[:, t], s, enc, src_mask)
            logits.append(logit_t)
            attns.append(alpha)
        return torch.stack(logits, dim=1), torch.stack(attns, dim=1)  # (B, T_tgt, V), (B, T_tgt, T_src)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_id: int, max_len: int) -> torch.Tensor:
        """Free-running decoding: feed back the argmax. Returns (B, max_len) token ids."""
        enc, s, src_mask = self.encode(src)
        y = torch.full((src.shape[0],), bos_id, dtype=torch.long, device=src.device)  # (B,)
        out = []
        for _ in range(max_len):
            logits, s, _ = self.decode_step(y, s, enc, src_mask)  # (B, V)
            y = logits.argmax(dim=-1)  # (B,)
            out.append(y)
        return torch.stack(out, dim=1)  # (B, max_len)


@torch.no_grad()
def beam_search(model: Seq2SeqAttention, src: torch.Tensor, bos_id: int, eos_id: int, max_len: int, beam: int = 3, length_alpha: float = 0.6) -> list[int]:
    """Beam search for ONE source sequence (src: (1, T_src)).

    Keeps the ``beam`` highest log-prob prefixes; finished hypotheses are scored with
    GNMT-style length normalisation  lp = ((5 + L) / 6) ** alpha.
    Returns the best token list (without BOS, up to and excluding EOS).
    """
    enc, s0, src_mask = model.encode(src)
    hyps: list[tuple[float, list[int], torch.Tensor]] = [(0.0, [bos_id], s0)]  # (logprob, tokens, state)
    finished: list[tuple[float, list[int]]] = []
    for _ in range(max_len):
        candidates: list[tuple[float, list[int], torch.Tensor]] = []
        for lp, toks, s in hyps:
            y = torch.tensor([toks[-1]], device=src.device)  # (1,)
            logits, s_new, _ = model.decode_step(y, s, enc, src_mask)  # (1, V)
            logp = F.log_softmax(logits, dim=-1)[0]  # (V,)
            top_lp, top_ix = logp.topk(beam)  # (beam,), (beam,)
            for v_lp, v in zip(top_lp.tolist(), top_ix.tolist()):
                candidates.append((lp + v_lp, toks + [v], s_new))
        candidates.sort(key=lambda c: c[0], reverse=True)
        hyps = []
        for lp, toks, s in candidates[:beam]:
            if toks[-1] == eos_id:
                L = len(toks) - 1
                finished.append((lp / (((5.0 + L) / 6.0) ** length_alpha), toks))
            else:
                hyps.append((lp, toks, s))
        if not hyps:
            break
    for lp, toks, _ in hyps:  # unfinished beams compete too
        L = len(toks) - 1
        finished.append((lp / (((5.0 + L) / 6.0) ** length_alpha), toks))
    best = max(finished, key=lambda f: f[0])[1]
    body = best[1:]
    return body[: body.index(eos_id)] if eos_id in body else body
