# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/interp/logit_lens.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k logit_lens -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py interp/logit_lens --force

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
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class Block(nn.Module):

    def __init__(self, d: int):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyResidualLM(nn.Module):
    """Pre-norm decoder: tokens (B, T) -> logits (B, T, V). Exposes ``residuals``."""

    def __init__(self, vocab: int, d: int=32, n_layers: int=3, max_len: int=32):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def residuals(self, tokens: torch.Tensor) -> list[torch.Tensor]:
        """h_0 .. h_L, each (B, T, d)."""
        raise NotImplementedError('TODO: implement residuals (see the reference in src/mlbook)')

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def logit_lens(model: TinyResidualLM, tokens: torch.Tensor) -> torch.Tensor:
    """Logits read from every residual position: (L + 1, B, T, V)."""
    raise NotImplementedError('TODO: implement logit_lens (see the reference in src/mlbook)')

def lens_trajectory(model: TinyResidualLM, tokens: torch.Tensor, position: int, answer: int) -> torch.Tensor:
    """Probability of ``answer`` at ``position`` after each layer: (L + 1,)."""
    raise NotImplementedError('TODO: implement lens_trajectory (see the reference in src/mlbook)')

def train_copy_task(model: TinyResidualLM, vocab: int, T: int, steps: int=300, seed: int=0) -> float:
    """Train to predict token[t-2] at position t (an induction-like 'copy from 2 back')."""
    raise NotImplementedError('TODO: implement train_copy_task (see the reference in src/mlbook)')
