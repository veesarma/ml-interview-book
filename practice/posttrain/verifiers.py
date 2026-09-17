# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/verifiers.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k verifiers -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/verifiers --force

"""Programmatic rewards ("verifiers") for the toy policy.

* ``arithmetic_messages`` / ``arithmetic_answer`` build single-digit addition prompts
  such as ``"3 + 4 ="`` whose verifiable answer is the token ``"7"``.
* ``arithmetic_reward`` returns 1.0 if the decoded response equals the answer, else 0.0
  (a *verifiable* reward for RLVR / GRPO).
* ``target_token_reward`` counts occurrences of a target token in a response (the
  programmatic reward used for the PPO demo: "say the target token as often as you can").
* ``make_arithmetic_sft_examples`` builds a *noisy* SFT set so the policy starts
  partially correct and RL has something to sharpen.
"""
from __future__ import annotations
import random
import torch
from mlbook.posttrain.chat_template import ToyTokenizer, prompt_ids, tokenize_chat
SYSTEM_PROMPT = 'you are a helpful assistant . answer briefly .'

def arithmetic_messages(a: int, b: int) -> list[dict[str, str]]:
    """System + user messages for the prompt ``"a + b ="``."""
    raise NotImplementedError('TODO: implement arithmetic_messages (see the reference in src/mlbook)')

def arithmetic_answer(a: int, b: int) -> str:
    """The single-token ground-truth answer (sums are kept below 10 so one digit suffices)."""
    raise NotImplementedError('TODO: implement arithmetic_answer (see the reference in src/mlbook)')

def all_arithmetic_pairs(max_sum: int=9) -> list[tuple[int, int]]:
    """Every ``(a, b)`` with ``0 <= a, b <= 9`` and ``a + b <= max_sum``."""
    raise NotImplementedError('TODO: implement all_arithmetic_pairs (see the reference in src/mlbook)')

def response_text(tokens: torch.Tensor, response_mask: torch.Tensor, tok: ToyTokenizer) -> str:
    """Decode the response tokens (mask == 1) of one row, dropping specials. Inputs (T,)."""
    raise NotImplementedError('TODO: implement response_text (see the reference in src/mlbook)')

def arithmetic_reward(tokens: torch.Tensor, response_mask: torch.Tensor, answers: list[str], tok: ToyTokenizer) -> torch.Tensor:
    """Binary verifiable reward: 1.0 iff the decoded response equals the expected answer.

    Args:
        tokens: (B, T) prompt + sampled response.
        response_mask: (B, T) 1.0 on response tokens.
        answers: list of length B with the expected answer strings.
    Returns:
        (B,) float rewards in {0, 1}.
    """
    raise NotImplementedError('TODO: implement arithmetic_reward (see the reference in src/mlbook)')

def target_token_reward(tokens: torch.Tensor, response_mask: torch.Tensor, target_id: int) -> torch.Tensor:
    """Count of ``target_id`` among response tokens, per row. ``tokens``/``mask`` (B, T) -> (B,)."""
    raise NotImplementedError('TODO: implement target_token_reward (see the reference in src/mlbook)')

def make_arithmetic_sft_examples(tok: ToyTokenizer, noise_rate: float, seed: int=0, repeats: int=4) -> list[tuple[list[int], list[int]]]:
    """Noisy SFT data: with probability ``noise_rate`` the labelled answer is a random digit.

    Returns a list of ``(input_ids, assistant_mask)`` ready for :func:`mlbook.posttrain.sft.pack_examples`.
    """
    raise NotImplementedError('TODO: implement make_arithmetic_sft_examples (see the reference in src/mlbook)')

def arithmetic_prompt_batch(pairs: list[tuple[int, int]], tok: ToyTokenizer) -> tuple[torch.Tensor, list[str]]:
    """Stack the prompts of ``pairs`` into one (B, T_p) tensor (all equal length) with their answers."""
    raise NotImplementedError('TODO: implement arithmetic_prompt_batch (see the reference in src/mlbook)')
