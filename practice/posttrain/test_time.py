# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/test_time.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k test_time -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/test_time --force

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
    raise NotImplementedError('TODO: implement bon_kl_bound (see the reference in src/mlbook)')

def best_of_n(model: nn.Module, prompt: torch.Tensor, n: int, reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], eos_id: int, max_new_tokens: int, temperature: float=1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``n`` responses to one prompt and return the highest-reward one.

    Args:
        prompt: (1, T_p) or (T_p,) token ids.
        reward_fn: ``(tokens (N, T), response_mask (N, T)) -> (N,)`` rewards.
    Returns:
        best tokens (T,), best mask (T,), all rewards (N,).
    """
    raise NotImplementedError('TODO: implement best_of_n (see the reference in src/mlbook)')

def self_consistency(model: nn.Module, prompt: torch.Tensor, n: int, extract_answer: Callable[[torch.Tensor, torch.Tensor], str], eos_id: int, max_new_tokens: int, temperature: float=1.0) -> tuple[str, Counter]:
    """Majority vote over ``n`` sampled answers. Returns ``(winning answer, vote counts)``."""
    raise NotImplementedError('TODO: implement self_consistency (see the reference in src/mlbook)')

@torch.no_grad()
def beam_search(model: nn.Module, prompt: torch.Tensor, beam_width: int, max_new_tokens: int, eos_id: int) -> list[tuple[list[int], float]]:
    """Length-unnormalised beam search over next tokens.

    Args:
        prompt: (T_p,) token ids.
    Returns:
        up to ``beam_width`` ``(token_list, log_prob)`` pairs sorted by log-prob, best first;
        ``token_list`` contains only the generated tokens (including a terminating ``eos_id``).
    """
    raise NotImplementedError('TODO: implement beam_search (see the reference in src/mlbook)')

def expected_max_of_n_gaussian(n: int, num_mc: int=20000, seed: int=0) -> float:
    """Monte-Carlo ``E[max of N standard normals]`` (the expected best-of-N reward under a Gaussian RM)."""
    raise NotImplementedError('TODO: implement expected_max_of_n_gaussian (see the reference in src/mlbook)')
