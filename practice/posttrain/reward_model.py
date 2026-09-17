# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/reward_model.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k reward_model -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/reward_model --force

"""Reward models and the Bradley-Terry pairwise loss.

A reward model ``r_phi(x, y)`` is an LM backbone with a scalar head read at the
*last* token of the (prompt, response) sequence. Preferences ``(x, y_w, y_l)`` are
fitted with the Bradley-Terry likelihood

    P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l)),

whose negative log-likelihood is ``-log sigma(r_w - r_l)``: logistic regression on
the reward *difference* with a fixed weight of 1 and no bias.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig

class TinyRewardModel(nn.Module):
    """LM backbone + scalar head on the last non-pad token. ``forward`` returns (B,) rewards."""

    def __init__(self, cfg: ToyLMConfig) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Scalar reward per sequence.

        Args:
            tokens: (B, T) prompt + response, right-padded.
            lengths: (B,) number of real tokens per row (>= 1).
        Returns:
            (B,) rewards ``r_phi(x, y)``.
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def bradley_terry_prob(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """``P(y_w > y_l) = sigma(r_w - r_l)``. Inputs and output (B,)."""
    raise NotImplementedError('TODO: implement bradley_terry_prob (see the reference in src/mlbook)')

def bradley_terry_loss(r_w: torch.Tensor, r_l: torch.Tensor, margin: float=0.0) -> torch.Tensor:
    """Pairwise loss ``-log sigma(r_w - r_l - margin)`` averaged over the batch.

    ``margin > 0`` (Llama 2 style) demands the gap exceed ``margin`` before the loss saturates.

    Args:
        r_w: (B,) rewards of the preferred responses.
        r_l: (B,) rewards of the rejected responses.
    Returns:
        scalar loss.
    """
    raise NotImplementedError('TODO: implement bradley_terry_loss (see the reference in src/mlbook)')

def pairwise_accuracy(r_w: torch.Tensor, r_l: torch.Tensor) -> float:
    """Fraction of pairs where the preferred response scores higher. Inputs (B,)."""
    raise NotImplementedError('TODO: implement pairwise_accuracy (see the reference in src/mlbook)')

def train_reward_model(rm: TinyRewardModel, chosen: torch.Tensor, chosen_len: torch.Tensor, rejected: torch.Tensor, rejected_len: torch.Tensor, steps: int=100, lr: float=0.003, batch_size: int=32, margin: float=0.0) -> list[float]:
    """Fit the Bradley-Terry loss on a fixed set of preference pairs (all tensors (N, T) / (N,)).

    Returns:
        per-step losses.
    """
    raise NotImplementedError('TODO: implement train_reward_model (see the reference in src/mlbook)')
