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
        super().__init__()
        self.backbone = TinyCausalLM(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Scalar reward per sequence.

        Args:
            tokens: (B, T) prompt + response, right-padded.
            lengths: (B,) number of real tokens per row (>= 1).
        Returns:
            (B,) rewards ``r_phi(x, y)``.
        """
        h = self.backbone.hidden_states(tokens)  # (B, T, d)
        last_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])  # (B, 1, d)
        h_last = h.gather(dim=1, index=last_idx).squeeze(1)  # (B, d) hidden state at last real token
        return self.head(h_last).squeeze(-1)  # (B,)


def bradley_terry_prob(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """``P(y_w > y_l) = sigma(r_w - r_l)``. Inputs and output (B,)."""
    return torch.sigmoid(r_w - r_l)  # (B,)


def bradley_terry_loss(r_w: torch.Tensor, r_l: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Pairwise loss ``-log sigma(r_w - r_l - margin)`` averaged over the batch.

    ``margin > 0`` (Llama 2 style) demands the gap exceed ``margin`` before the loss saturates.

    Args:
        r_w: (B,) rewards of the preferred responses.
        r_l: (B,) rewards of the rejected responses.
    Returns:
        scalar loss.
    """
    return -F.logsigmoid(r_w - r_l - margin).mean()  # scalar


def pairwise_accuracy(r_w: torch.Tensor, r_l: torch.Tensor) -> float:
    """Fraction of pairs where the preferred response scores higher. Inputs (B,)."""
    return float((r_w > r_l).float().mean())


def train_reward_model(
    rm: TinyRewardModel,
    chosen: torch.Tensor,
    chosen_len: torch.Tensor,
    rejected: torch.Tensor,
    rejected_len: torch.Tensor,
    steps: int = 100,
    lr: float = 3e-3,
    batch_size: int = 32,
    margin: float = 0.0,
) -> list[float]:
    """Fit the Bradley-Terry loss on a fixed set of preference pairs (all tensors (N, T) / (N,)).

    Returns:
        per-step losses.
    """
    opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.0)
    N = chosen.shape[0]
    losses: list[float] = []
    rm.train()
    for step in range(steps):
        idx = torch.randint(0, N, (batch_size,))  # (B,)
        r_w = rm(chosen[idx], chosen_len[idx])  # (B,)
        r_l = rm(rejected[idx], rejected_len[idx])  # (B,)
        loss = bradley_terry_loss(r_w, r_l, margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
