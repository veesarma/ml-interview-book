# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/posttrain/dpo.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k dpo -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py posttrain/dpo --force

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

def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss and the implicit rewards ``beta * log(pi / pi_ref)``.

    Args:
        pi_chosen, pi_rejected: (B,) ``log pi_theta(y_w|x)``, ``log pi_theta(y_l|x)``.
        ref_chosen, ref_rejected: (B,) the same under the frozen reference.
    Returns:
        loss (scalar), chosen_rewards (B,), rejected_rewards (B,).
    """
    raise NotImplementedError('TODO: implement dpo_loss (see the reference in src/mlbook)')

def ipo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, tau: float) -> torch.Tensor:
    """IPO: squared regression of the log-ratio gap onto ``1 / (2 tau)``. Inputs (B,), output scalar."""
    raise NotImplementedError('TODO: implement ipo_loss (see the reference in src/mlbook)')

def simpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, len_chosen: torch.Tensor, len_rejected: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    """SimPO: length-normalised log-prob margin with a target margin ``gamma``; no reference model.

    Args:
        pi_chosen, pi_rejected: (B,) summed log-probs under the policy.
        len_chosen, len_rejected: (B,) response lengths in tokens.
    """
    raise NotImplementedError('TODO: implement simpo_loss (see the reference in src/mlbook)')

def orpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, len_chosen: torch.Tensor, len_rejected: torch.Tensor, lam: float) -> torch.Tensor:
    """ORPO: SFT NLL on the chosen response plus an odds-ratio preference term (reference-free).

    ``odds(y) = p / (1 - p)`` with ``p = exp(mean token log-prob)``.
    """
    raise NotImplementedError('TODO: implement orpo_loss (see the reference in src/mlbook)')

def kto_reference_point(pi: torch.Tensor, ref: torch.Tensor, beta: float) -> torch.Tensor:
    """The KTO reference point ``z_0 = max(0, beta * mean[log pi - log pi_ref])``, detached.

    In the paper ``z_0`` estimates ``beta * KL(pi || pi_ref)`` from *mismatched* ``(x, y')`` pairs
    (a response shuffled to a different prompt), so that it measures how far the policy has moved
    overall rather than how good these particular examples are. Passing those mismatched log-probs
    here reproduces that estimator; passing the batch's own log-probs gives the cheap in-batch
    approximation. It is clamped at 0 and never back-propagated through.

    Args:
        pi, ref: (B,) summed log-probs under policy / reference.
    Returns:
        scalar reference point.
    """
    raise NotImplementedError('TODO: implement kto_reference_point (see the reference in src/mlbook)')

def kto_loss(pi: torch.Tensor, ref: torch.Tensor, desirable: torch.Tensor, beta: float, lambda_d: float=1.0, lambda_u: float=1.0, z0: torch.Tensor | float | None=None) -> torch.Tensor:
    """KTO on *unpaired* examples: each response is labelled desirable (1) or undesirable (0).

    ``L = mean[lambda_y - v]`` with ``v = lambda_D sigma(beta(r - z0))`` for desirable examples and
    ``v = lambda_U sigma(z0 - beta r)`` for undesirable ones, where ``r = beta(log pi - log pi_ref)``
    is the implicit reward. A desirable example is rewarded for sitting *above* the reference point
    ``z0`` and an undesirable one for sitting below it; ``lambda_U > lambda_D`` encodes loss aversion.

    Args:
        pi, ref: (B,) summed log-probs under policy / reference.
        desirable: (B,) float in {0, 1}.
        z0: the reference point. Defaults to :func:`kto_reference_point` on this batch, which is
            degenerate when every example has the same implicit reward (it then equals that reward);
            pass an explicit estimate from mismatched pairs to reproduce the paper.
    Returns:
        scalar loss.
    """
    raise NotImplementedError('TODO: implement kto_loss (see the reference in src/mlbook)')

def dpo_train_step(policy: nn.Module, ref: nn.Module, opt: torch.optim.Optimizer, chosen: torch.Tensor, chosen_mask: torch.Tensor, rejected: torch.Tensor, rejected_mask: torch.Tensor, beta: float) -> tuple[float, float]:
    """One DPO gradient step on a batch of (chosen, rejected) sequences, all (B, T).

    Returns ``(loss, mean implicit reward margin)``.
    """
    raise NotImplementedError('TODO: implement dpo_train_step (see the reference in src/mlbook)')
