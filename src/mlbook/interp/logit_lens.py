"""A tiny residual-stream language model plus the logit lens (nostalgebraist 2020).

Residual stream: h_0 = E[tokens] + pos;  h_{l+1} = h_l + Attn_l(LN(h_l)) + MLP_l(LN(h_l))
Logit lens: read the residual stream after every layer with the *final* norm and
unembedding: logits_l = LN_f(h_l) W_U. Because every block only *adds* to h, the
model's intermediate guess is well defined at every layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    """Single-head causal attention with three explicit projections. (B, T, d) -> (B, T, d)."""

    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)  # each (B, T, d)
        scores = Q @ K.transpose(1, 2) / d**0.5  # (B, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # (T, T)
        scores = scores.masked_fill(mask, float("-inf"))  # (B, T, T)
        return self.o(torch.softmax(scores, dim=-1) @ V)  # (B, T, d)


class Block(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.ln1(h))  # (B, T, d)
        return h + self.fc2(F.gelu(self.fc1(self.ln2(h))))  # (B, T, d)


class TinyResidualLM(nn.Module):
    """Pre-norm decoder: tokens (B, T) -> logits (B, T, V). Exposes ``residuals``."""

    def __init__(self, vocab: int, d: int = 32, n_layers: int = 3, max_len: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, vocab, bias=False)

    def residuals(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """h_0 .. h_L, each (B, T, d)."""
        B, T = tokens.shape
        h = self.embed(tokens) + self.pos(torch.arange(T))  # (B, T, d)
        hs = [h]
        for block in self.blocks:
            h = block(h)  # (B, T, d)
            hs.append(h)
        return hs

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.ln_f(self.residuals(tokens)[-1]))  # (B, T, V)


def logit_lens(model: TinyResidualLM, tokens: torch.Tensor) -> torch.Tensor:
    """Logits read from every residual position: (L + 1, B, T, V)."""
    with torch.no_grad():
        return torch.stack([model.unembed(model.ln_f(h)) for h in model.residuals(tokens)])  # (L+1, B, T, V)


def lens_trajectory(model: TinyResidualLM, tokens: torch.Tensor, position: int, answer: int) -> torch.Tensor:
    """Probability of ``answer`` at ``position`` after each layer: (L + 1,)."""
    lens = logit_lens(model, tokens)  # (L+1, B, T, V)
    return torch.softmax(lens[:, 0, position], dim=-1)[:, answer]  # (L+1,)


def train_copy_task(model: TinyResidualLM, vocab: int, T: int, steps: int = 300, seed: int = 0) -> float:
    """Train to predict token[t-2] at position t (an induction-like 'copy from 2 back')."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tokens = torch.randint(0, vocab, (16, T), generator=g)  # (B, T)
        logits = model(tokens)[:, 2:]  # (B, T-2, V) predictions at positions 2..T-1
        loss = F.cross_entropy(logits.reshape(-1, vocab), tokens[:, :-2].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())
