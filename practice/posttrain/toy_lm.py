# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/toy_lm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k toy_lm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/toy_lm --force

"""A tiny, explicit causal language model used as the *policy* throughout Part VII.

The model is deliberately small (a few thousand parameters) so that SFT, reward
modelling, PPO, DPO and GRPO all train on a CPU in seconds. It is a standard
pre-norm decoder: token + learned positional embeddings, ``n_layers`` blocks of
causal self-attention (three separate Q/K/V projections) and an MLP, then a tied
LM head. It also exposes the utilities every post-training algorithm needs:

* ``token_log_probs``: per-token log pi(y_t | y_<t) from logits,
* ``sequence_log_prob``: the sum of those over a response mask,
* ``sample``: autoregressive sampling with temperature and an EOS token.

Shape vocabulary: ``B`` batch, ``T`` sequence length, ``V`` vocab, ``d`` model
width, ``H`` heads, ``d_head = d // H``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ToyLMConfig:
    """Hyper-parameters of :class:`TinyCausalLM`."""
    vocab_size: int
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 2
    max_len: int = 32
    dropout: float = 0.0

def causal_mask(T: int, device: torch.device | None=None) -> torch.Tensor:
    """Lower-triangular boolean mask: ``mask[i, j] = True`` iff position ``i`` may attend to ``j <= i``.

    Returns:
        (T, T) bool tensor.
    """
    raise NotImplementedError('TODO: implement causal_mask (see the reference in src/mlbook)')

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with three explicit projections.

    Input ``x``: (B, T, d). Output: (B, T, d).
    Equation: ``softmax(Q K^T / sqrt(d_head) + mask) V`` per head, heads concatenated.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """``allowed`` is a (B, T, T) or (T, T) boolean mask of permitted attention edges."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class Block(nn.Module):
    """Pre-norm Transformer block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class TinyCausalLM(nn.Module):
    """Decoder-only LM. ``forward(tokens)`` returns next-token logits of shape (B, T, V)."""

    def __init__(self, cfg: ToyLMConfig) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def hidden_states(self, tokens: torch.Tensor, allowed: torch.Tensor | None=None) -> torch.Tensor:
        """Final hidden states before the LM head.

        Args:
            tokens: (B, T) int64 token ids.
            allowed: optional (B, T, T) bool attention mask (used for packed SFT);
                defaults to the plain causal mask.
        Returns:
            (B, T, d) hidden states.
        """
        raise NotImplementedError('TODO: implement hidden_states (see the reference in src/mlbook)')

    def forward(self, tokens: torch.Tensor, allowed: torch.Tensor | None=None) -> torch.Tensor:
        """Next-token logits: ``logits[:, t]`` predicts ``tokens[:, t + 1]``. Returns (B, T, V)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def token_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities ``log pi(tokens[:, t] | tokens[:, :t])`` for ``t >= 1``.

    Args:
        logits: (B, T, V) next-token logits from :class:`TinyCausalLM`.
        tokens: (B, T) the sequence that produced those logits.
    Returns:
        (B, T-1) log-probs; entry ``t`` scores ``tokens[:, t + 1]`` given the prefix.
    """
    raise NotImplementedError('TODO: implement token_log_probs (see the reference in src/mlbook)')

def sequence_log_prob(model: nn.Module, tokens: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """``log pi(y | x) = sum_t mask_t * log pi(tokens_t | tokens_<t)`` over response tokens.

    Args:
        model: a :class:`TinyCausalLM`.
        tokens: (B, T) prompt followed by response (and padding).
        response_mask: (B, T) 1.0 on response tokens (targets to score), 0.0 elsewhere.
    Returns:
        (B,) summed response log-probability.
    """
    raise NotImplementedError('TODO: implement sequence_log_prob (see the reference in src/mlbook)')

@torch.no_grad()
def sample(model: nn.Module, prompt: torch.Tensor, max_new_tokens: int, eos_id: int, temperature: float=1.0, greedy: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively sample ``max_new_tokens`` continuations of ``prompt``.

    Generation continues for every row until it emits ``eos_id``; afterwards the
    row is padded with ``eos_id`` and masked out.

    Args:
        prompt: (B, T_p) prompt token ids (all rows the same length).
        max_new_tokens: number of response positions to generate.
    Returns:
        tokens: (B, T_p + max_new_tokens) prompt + response (+ eos padding).
        response_mask: (B, T_p + max_new_tokens) float, 1.0 on generated tokens up to
            and including the first eos, 0.0 on the prompt and on padding after eos.
    """
    raise NotImplementedError('TODO: implement sample (see the reference in src/mlbook)')
