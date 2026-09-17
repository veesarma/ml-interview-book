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


def causal_mask(T: int, device: torch.device | None = None) -> torch.Tensor:
    """Lower-triangular boolean mask: ``mask[i, j] = True`` iff position ``i`` may attend to ``j <= i``.

    Returns:
        (T, T) bool tensor.
    """
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))  # (T, T)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with three explicit projections.

    Input ``x``: (B, T, d). Output: (B, T, d).
    Equation: ``softmax(Q K^T / sqrt(d_head) + mask) V`` per head, heads concatenated.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """``allowed`` is a (B, T, T) or (T, T) boolean mask of permitted attention edges."""
        B, T, d = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)
        scores = q @ k.transpose(-2, -1) / (self.d_head ** 0.5)  # (B, H, T, T)
        if allowed.dim() == 2:
            allowed = allowed.unsqueeze(0)  # (1, T, T)
        scores = scores.masked_fill(~allowed.unsqueeze(1), float("-inf"))  # (B, H, T, T)
        weights = self.dropout(F.softmax(scores, dim=-1))  # (B, H, T, T)
        ctx = weights @ v  # (B, H, T, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, d)  # (B, T, d)
        return self.out_proj(ctx)  # (B, T, d)


class Block(nn.Module):
    """Pre-norm Transformer block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc_in = nn.Linear(d_model, 4 * d_model)
        self.fc_out = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), allowed)  # (B, T, d)
        h = self.fc_out(F.gelu(self.fc_in(self.ln2(x))))  # (B, T, d)
        return x + self.dropout(h)  # (B, T, d)


class TinyCausalLM(nn.Module):
    """Decoder-only LM. ``forward(tokens)`` returns next-token logits of shape (B, T, V)."""

    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying: (V, d) shared
        nn.init.normal_(self.tok_emb.weight, std=0.02)  # small init keeps initial logits near uniform
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def hidden_states(self, tokens: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
        """Final hidden states before the LM head.

        Args:
            tokens: (B, T) int64 token ids.
            allowed: optional (B, T, T) bool attention mask (used for packed SFT);
                defaults to the plain causal mask.
        Returns:
            (B, T, d) hidden states.
        """
        B, T = tokens.shape
        if T > self.cfg.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.cfg.max_len}")
        if allowed is None:
            allowed = causal_mask(T, tokens.device)  # (T, T)
        pos = torch.arange(T, device=tokens.device)  # (T,)
        x = self.tok_emb(tokens) + self.pos_emb(pos)  # (B, T, d)
        for block in self.blocks:
            x = block(x, allowed)  # (B, T, d)
        return self.ln_f(x)  # (B, T, d)

    def forward(self, tokens: torch.Tensor, allowed: torch.Tensor | None = None) -> torch.Tensor:
        """Next-token logits: ``logits[:, t]`` predicts ``tokens[:, t + 1]``. Returns (B, T, V)."""
        return self.lm_head(self.hidden_states(tokens, allowed))  # (B, T, V)


def token_log_probs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities ``log pi(tokens[:, t] | tokens[:, :t])`` for ``t >= 1``.

    Args:
        logits: (B, T, V) next-token logits from :class:`TinyCausalLM`.
        tokens: (B, T) the sequence that produced those logits.
    Returns:
        (B, T-1) log-probs; entry ``t`` scores ``tokens[:, t + 1]`` given the prefix.
    """
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    targets = tokens[:, 1:].unsqueeze(-1)  # (B, T-1, 1)
    return logp.gather(dim=-1, index=targets).squeeze(-1)  # (B, T-1)


def sequence_log_prob(model: nn.Module, tokens: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """``log pi(y | x) = sum_t mask_t * log pi(tokens_t | tokens_<t)`` over response tokens.

    Args:
        model: a :class:`TinyCausalLM`.
        tokens: (B, T) prompt followed by response (and padding).
        response_mask: (B, T) 1.0 on response tokens (targets to score), 0.0 elsewhere.
    Returns:
        (B,) summed response log-probability.
    """
    logits = model(tokens)  # (B, T, V)
    logp = token_log_probs(logits, tokens)  # (B, T-1) scores tokens[:, 1:]
    mask = response_mask[:, 1:].to(logp.dtype)  # (B, T-1) aligned with logp
    return (logp * mask).sum(dim=-1)  # (B,)


@torch.no_grad()
def sample(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    greedy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    B, T_p = prompt.shape
    tokens = prompt.clone()  # (B, T_p)
    alive = torch.ones(B, dtype=torch.bool, device=prompt.device)  # (B,) still generating
    mask_cols = [torch.zeros(B, T_p, device=prompt.device)]  # list of (B, *) mask pieces
    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :] / temperature  # (B, V) last-position logits
        if greedy:
            next_tok = logits.argmax(dim=-1)  # (B,)
        else:
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)  # (B,)
        next_tok = torch.where(alive, next_tok, torch.full_like(next_tok, eos_id))  # (B,)
        mask_cols.append(alive.float().unsqueeze(1))  # (B, 1)
        tokens = torch.cat([tokens, next_tok.unsqueeze(1)], dim=1)  # (B, T_p + k)
        alive = alive & (next_tok != eos_id)  # (B,)
    response_mask = torch.cat(mask_cols, dim=1)  # (B, T_p + max_new_tokens)
    return tokens, response_mask
