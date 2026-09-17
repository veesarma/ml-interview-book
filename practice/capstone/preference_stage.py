# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/preference_stage.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k preference_stage -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/preference_stage --force

"""Canon #60, part 7 -- DPO and a simplified GRPO, each as one small function.

Both algorithms optimise the same KL-regularised objective

``max_pi E_{y ~ pi}[r(x, y)] - beta * KL(pi || pi_ref)``

and they differ in what stands in for ``r``.

DPO never builds a reward model. The optimum of the objective above is
``pi*(y|x) = pi_ref(y|x) exp(r(x,y)/beta) / Z(x)``; invert it for ``r``,
substitute into Bradley-Terry, and ``Z(x)`` cancels because it depends only on
``x``. What is left is a classification loss on the difference of two log
ratios, computable with two forward passes of the policy and two of the frozen
reference.

GRPO keeps the RL loop but drops the critic. Sample a group of ``G`` responses
per prompt, score each with the verifier, and use the group-normalised reward as
every token's advantage. The baseline is the group mean, which is what makes the
advantages sum to zero within a group and is why GRPO needs no value network.

Shapes: ``P`` prompts per step, ``G`` samples per prompt, ``B = P * G`` rows.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F
from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM
from mlbook.capstone.reward_stage import PreferencePair, verifier_rewards
from mlbook.capstone.sft_stage import SFTBatch, encode_batch, next_token_logits

def freeze_reference(model: TinyVLM) -> TinyVLM:
    """Deep-copy a policy and freeze it. The result is ``pi_ref`` for DPO and GRPO."""
    raise NotImplementedError('TODO: implement freeze_reference (see the reference in src/mlbook)')

def sequence_logprob(model: TinyVLM, batch: SFTBatch) -> torch.Tensor:
    """Summed log-probability of the assistant tokens. Returns ``(B,)``.

    ``log pi(y | x) = sum_t log pi(y_t | x, y_<t)``, restricted to the positions
    ``loss_mask`` marks. Prompt tokens contribute nothing, which is what makes
    the DPO log ratio a property of the response.
    """
    raise NotImplementedError('TODO: implement sequence_logprob (see the reference in src/mlbook)')

def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float=0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """The DPO objective. All inputs ``(B,)`` summed log-probabilities.

    Returns ``(loss, margin)`` where

    ``margin = (log pi(y_w) - log pi_ref(y_w)) - (log pi(y_l) - log pi_ref(y_l))``

    is the difference of implicit rewards divided by ``beta``, and

    ``loss = -log sigma(beta * margin)``.

    Training drives ``margin`` up. Watch it rather than the loss: the loss also
    falls when the policy pushes ``log pi(y_l)`` down without raising
    ``log pi(y_w)``, which is the well-known DPO failure of degrading both.
    """
    raise NotImplementedError('TODO: implement dpo_loss (see the reference in src/mlbook)')

@torch.no_grad()
def mean_dpo_margin(policy: TinyVLM, reference: TinyVLM, pairs: list[PreferencePair]) -> float:
    """Mean implicit-reward margin over a set of pairs, in nats."""
    raise NotImplementedError('TODO: implement mean_dpo_margin (see the reference in src/mlbook)')

def run_dpo(policy: TinyVLM, reference: TinyVLM, pairs: list[PreferencePair], steps: int=120, batch_size: int=32, lr: float=0.0003, beta: float=0.1, seed: int=0) -> dict[str, object]:
    """Run DPO on the frozen reference. Returns the loss curve and the margin before/after."""
    raise NotImplementedError('TODO: implement run_dpo (see the reference in src/mlbook)')

def group_advantages(rewards: torch.Tensor, normalize_std: bool=True, eps: float=1e-06) -> torch.Tensor:
    """``A_i = (r_i - mean_g) / (std_g + eps)``. Input and output ``(P, G)``.

    Subtracting the group mean is the whole trick: it is a baseline that depends
    on the prompt but not on the sampled response, so it leaves the policy
    gradient unbiased while removing the per-prompt difficulty. Every row sums
    to zero by construction, whether or not the std is divided out. Dividing by
    the std reweights easy and hard prompts; Dr. GRPO drops it for that reason.
    """
    raise NotImplementedError('TODO: implement group_advantages (see the reference in src/mlbook)')

def k3_kl(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Schulman's ``k3`` estimator of ``KL(pi || pi_ref)``: ``exp(q - p) - (q - p) - 1``.

    Inputs ``(B,)`` log-probabilities under the policy (``p``) and the reference
    (``q``). Non-negative for every sample and unbiased in expectation, unlike
    the raw ``p - q`` difference which is unbiased but can go negative and makes
    the loss curve unreadable.
    """
    raise NotImplementedError('TODO: implement k3_kl (see the reference in src/mlbook)')

def grpo_loss(logp: torch.Tensor, logp_old: torch.Tensor, advantages: torch.Tensor, logp_ref: torch.Tensor, clip_eps: float=0.2, kl_coef: float=0.02) -> torch.Tensor:
    """Clipped policy-gradient loss with a ``k3`` KL penalty. All inputs ``(B,)``, output scalar.

    ``ratio = exp(log pi - log pi_old)``; the surrogate is
    ``min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)``. On the first inner epoch
    the ratio is exactly 1 and this reduces to REINFORCE with a group baseline.
    """
    raise NotImplementedError('TODO: implement grpo_loss (see the reference in src/mlbook)')

@torch.no_grad()
def sample_group(model: TinyVLM, examples: list[task.Example], group_size: int, temperature: float=1.0, generator: torch.Generator | None=None) -> tuple[list[task.Example], list[str], torch.Tensor]:
    """Sample ``G`` answers for each of ``P`` prompts.

    Returns ``(repeated_examples, sampled_answers, rewards)`` with
    ``len == P * G`` and ``rewards`` shaped ``(P, G)``.
    """
    raise NotImplementedError('TODO: implement sample_group (see the reference in src/mlbook)')

def run_grpo(policy: TinyVLM, reference: TinyVLM, examples: list[task.Example], steps: int=60, prompts_per_step: int=16, group_size: int=8, inner_epochs: int=2, lr: float=0.0002, clip_eps: float=0.2, kl_coef: float=0.02, temperature: float=1.0, seed: int=0) -> dict[str, object]:
    """Sample, score with the verifier, normalise within the group, update.

    Each step is one full on-policy iteration: generation, reward, advantage,
    then ``inner_epochs`` gradient steps on the same batch under the clip. That
    ordering is the part worth being able to recite.
    """
    raise NotImplementedError('TODO: implement run_grpo (see the reference in src/mlbook)')
