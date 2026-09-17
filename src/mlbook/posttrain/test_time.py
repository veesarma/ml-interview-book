"""Test-time compute: best-of-N, self-consistency (majority vote) and beam search over the toy LM."""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlbook.posttrain.toy_lm import sample


def bon_kl_bound(n: int) -> float:
    """``KL(BoN || pi) <= log N - (N - 1) / N``: the KL cost of best-of-N sampling."""
    return math.log(n) - (n - 1) / n


def best_of_n(
    model: nn.Module,
    prompt: torch.Tensor,
    n: int,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eos_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``n`` responses to one prompt and return the highest-reward one.

    Args:
        prompt: (1, T_p) or (T_p,) token ids.
        reward_fn: ``(tokens (N, T), response_mask (N, T)) -> (N,)`` rewards.
    Returns:
        best tokens (T,), best mask (T,), all rewards (N,).
    """
    prompt = prompt.view(1, -1).expand(n, -1)  # (N, T_p)
    tokens, mask = sample(model, prompt, max_new_tokens, eos_id, temperature)  # (N, T), (N, T)
    rewards = reward_fn(tokens, mask)  # (N,)
    best = int(rewards.argmax())
    return tokens[best], mask[best], rewards


def self_consistency(
    model: nn.Module,
    prompt: torch.Tensor,
    n: int,
    extract_answer: Callable[[torch.Tensor, torch.Tensor], str],
    eos_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> tuple[str, Counter]:
    """Majority vote over ``n`` sampled answers. Returns ``(winning answer, vote counts)``."""
    prompt = prompt.view(1, -1).expand(n, -1)  # (N, T_p)
    tokens, mask = sample(model, prompt, max_new_tokens, eos_id, temperature)  # (N, T), (N, T)
    votes = Counter(extract_answer(tokens[i], mask[i]) for i in range(n))
    return votes.most_common(1)[0][0], votes


@torch.no_grad()
def beam_search(
    model: nn.Module, prompt: torch.Tensor, beam_width: int, max_new_tokens: int, eos_id: int
) -> list[tuple[list[int], float]]:
    """Length-unnormalised beam search over next tokens.

    Args:
        prompt: (T_p,) token ids.
    Returns:
        up to ``beam_width`` ``(token_list, log_prob)`` pairs sorted by log-prob, best first;
        ``token_list`` contains only the generated tokens (including a terminating ``eos_id``).
    """
    beams: list[tuple[list[int], float]] = [([], 0.0)]  # (generated tokens, cumulative log-prob)
    finished: list[tuple[list[int], float]] = []
    for _ in range(max_new_tokens):
        candidates: list[tuple[list[int], float]] = []
        for gen, score in beams:
            seq = torch.cat([prompt, torch.tensor(gen, dtype=prompt.dtype)]).unsqueeze(0)  # (1, T_p + len(gen))
            logp = F.log_softmax(model(seq)[0, -1], dim=-1)  # (V,) next-token log-probs
            top_logp, top_ids = logp.topk(beam_width)  # (beam_width,) each
            for lp, tid in zip(top_logp.tolist(), top_ids.tolist()):
                candidates.append((gen + [tid], score + lp))
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for gen, score in candidates[:beam_width]:
            (finished if gen[-1] == eos_id else beams).append((gen, score))
        if not beams:
            break
    finished.extend(beams)  # unfinished beams count too
    finished.sort(key=lambda c: c[1], reverse=True)
    return finished[:beam_width]


def expected_max_of_n_gaussian(n: int, num_mc: int = 20000, seed: int = 0) -> float:
    """Monte-Carlo ``E[max of N standard normals]`` (the expected best-of-N reward under a Gaussian RM)."""
    g = torch.Generator().manual_seed(seed)
    draws = torch.randn(num_mc, n, generator=g)  # (num_mc, N)
    return float(draws.max(dim=1).values.mean())
