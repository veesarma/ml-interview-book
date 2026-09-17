"""Direct preference optimisation and its relatives, as explicit functions of sequence log-probs.

Every loss below takes the *summed* response log-probabilities of the chosen and
rejected responses under the policy (and, where needed, the reference), each of
shape (B,), and returns a scalar. ``dpo_train_step`` wires them to the toy policy.

    DPO   : -log sigma(beta * [(pi_c - ref_c) - (pi_r - ref_r)])
    IPO   : ([(pi_c - ref_c) - (pi_r - ref_r)] - 1/(2 tau))^2
    SimPO : -log sigma(beta * [pi_c / |y_c| - pi_r / |y_r|] - gamma)      (no reference model)
    ORPO  : NLL(chosen) + lambda * -log sigma(log odds(chosen) - log odds(rejected))
    KTO   : unpaired; v = lambda_D sigma(beta (r - z0)) for desirable, lambda_U sigma(beta (z0 - r)) otherwise
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlbook.posttrain.toy_lm import sequence_log_prob


def dpo_loss(
    pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss and the implicit rewards ``beta * log(pi / pi_ref)``.

    Args:
        pi_chosen, pi_rejected: (B,) ``log pi_theta(y_w|x)``, ``log pi_theta(y_l|x)``.
        ref_chosen, ref_rejected: (B,) the same under the frozen reference.
    Returns:
        loss (scalar), chosen_rewards (B,), rejected_rewards (B,).
    """
    chosen_rewards = beta * (pi_chosen - ref_chosen)  # (B,) implicit reward of y_w
    rejected_rewards = beta * (pi_rejected - ref_rejected)  # (B,) implicit reward of y_l
    margin = chosen_rewards - rejected_rewards  # (B,)
    loss = -F.logsigmoid(margin).mean()  # scalar
    return loss, chosen_rewards.detach(), rejected_rewards.detach()


def ipo_loss(
    pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, tau: float
) -> torch.Tensor:
    """IPO: squared regression of the log-ratio gap onto ``1 / (2 tau)``. Inputs (B,), output scalar."""
    gap = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)  # (B,)
    return ((gap - 1.0 / (2.0 * tau)) ** 2).mean()  # scalar


def simpo_loss(
    pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, len_chosen: torch.Tensor, len_rejected: torch.Tensor,
    beta: float, gamma: float,
) -> torch.Tensor:
    """SimPO: length-normalised log-prob margin with a target margin ``gamma``; no reference model.

    Args:
        pi_chosen, pi_rejected: (B,) summed log-probs under the policy.
        len_chosen, len_rejected: (B,) response lengths in tokens.
    """
    avg_chosen = pi_chosen / len_chosen  # (B,) per-token log-prob of y_w
    avg_rejected = pi_rejected / len_rejected  # (B,)
    return -F.logsigmoid(beta * (avg_chosen - avg_rejected) - gamma).mean()  # scalar


def orpo_loss(
    pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, len_chosen: torch.Tensor, len_rejected: torch.Tensor, lam: float
) -> torch.Tensor:
    """ORPO: SFT NLL on the chosen response plus an odds-ratio preference term (reference-free).

    ``odds(y) = p / (1 - p)`` with ``p = exp(mean token log-prob)``.
    """
    avg_chosen = pi_chosen / len_chosen  # (B,) mean log-prob -> log p_w
    avg_rejected = pi_rejected / len_rejected  # (B,)
    log_odds_chosen = avg_chosen - torch.log1p(-torch.exp(avg_chosen).clamp(max=1 - 1e-6))  # (B,)
    log_odds_rejected = avg_rejected - torch.log1p(-torch.exp(avg_rejected).clamp(max=1 - 1e-6))  # (B,)
    nll = -avg_chosen.mean()  # scalar SFT term
    return nll - lam * F.logsigmoid(log_odds_chosen - log_odds_rejected).mean()  # scalar


def kto_loss(
    pi: torch.Tensor, ref: torch.Tensor, desirable: torch.Tensor, beta: float, lambda_d: float = 1.0, lambda_u: float = 1.0
) -> torch.Tensor:
    """KTO on *unpaired* examples: each response is labelled desirable (1) or undesirable (0).

    ``z0`` (the reference point) is the batch-mean implicit reward, clamped at 0 and detached.
    Args:
        pi, ref: (B,) summed log-probs under policy / reference.
        desirable: (B,) float in {0, 1}.
    """
    reward = beta * (pi - ref)  # (B,) implicit reward
    z0 = reward.mean().detach().clamp(min=0.0)  # scalar reference point
    value_d = lambda_d * torch.sigmoid(reward - z0)  # (B,) gain if desirable
    value_u = lambda_u * torch.sigmoid(z0 - reward)  # (B,) loss-aversion if undesirable
    value = desirable * value_d + (1.0 - desirable) * value_u  # (B,)
    weight = desirable * lambda_d + (1.0 - desirable) * lambda_u  # (B,)
    return (weight - value).mean()  # scalar


def dpo_train_step(
    policy: nn.Module,
    ref: nn.Module,
    opt: torch.optim.Optimizer,
    chosen: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float,
) -> tuple[float, float]:
    """One DPO gradient step on a batch of (chosen, rejected) sequences, all (B, T).

    Returns ``(loss, mean implicit reward margin)``.
    """
    with torch.no_grad():
        ref_c = sequence_log_prob(ref, chosen, chosen_mask)  # (B,)
        ref_r = sequence_log_prob(ref, rejected, rejected_mask)  # (B,)
    pi_c = sequence_log_prob(policy, chosen, chosen_mask)  # (B,)
    pi_r = sequence_log_prob(policy, rejected, rejected_mask)  # (B,)
    loss, r_c, r_r = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach()), float((r_c - r_r).mean())
