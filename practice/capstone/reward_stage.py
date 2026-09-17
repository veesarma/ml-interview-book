# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/reward_stage.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k reward_stage -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/reward_stage --force

"""Canon #60, part 6 -- two rewards: one learned, one computed.

Post-training needs a scalar that says how good a response was. There are two
ways to get one and this module implements both, because the comparison is the
interview question.

**Learned (Bradley-Terry).** Collect pairs ``(x, y_w, y_l)`` where a human
preferred ``y_w``. Fit a scalar head ``r_phi`` by maximum likelihood under
``P(y_w > y_l) = sigma(r_w - r_l)``, which gives the loss
``-log sigma(r_w - r_l)``. Only differences are identified: adding a constant to
every reward leaves the loss unchanged. Works for anything you can rank and
nothing else; it can be over-optimised, because the policy learns the head's
quirks along with its signal.

**Computed (a verifier).** Run a program on the response. Here the program
compares the answer token to the count produced by the scene generator, so the
reward is correct by construction and cannot be gamed by producing text the
reward model happens to like. This is the RLVR setting: it needs a task with a
checkable answer, which is why maths, code and counting dominate the RLVR
literature.

The two are complementary. The verifier is the ground truth this capstone
optimises against; the reward model is trained on the same pairs so you can
measure how well a learned proxy tracks it.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM
from mlbook.capstone.sft_stage import encode_batch, next_token_logits

def verifier_reward(prediction: str, example: task.Example) -> float:
    """Programmatic reward for one answer. 1.0 if it matches the computed answer, else 0.0.

    The gold answer comes from :func:`mlbook.capstone.synthetic_task.count_shapes`
    or :func:`~mlbook.capstone.synthetic_task.largest_colour` run on the scene
    graph that drew the image, so no annotation and no judge model is involved.
    """
    raise NotImplementedError('TODO: implement verifier_reward (see the reference in src/mlbook)')

def verifier_rewards(predictions: list[str], examples: list[task.Example]) -> torch.Tensor:
    """Vectorised :func:`verifier_reward`. Returns ``(B,)`` float32."""
    raise NotImplementedError('TODO: implement verifier_rewards (see the reference in src/mlbook)')

class TinyRewardModel(nn.Module):
    """A VLM trunk with a scalar head instead of a vocabulary head.

    ``score(images, input_ids, attention_mask)`` returns ``(B,)``: the value of
    the final hidden state at the last real token of the sequence, which is the
    standard reward-model head (read the end of the response, score the whole
    thing).
    """

    def __init__(self, trunk: TinyVLM) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @classmethod
    def from_policy(cls, policy: TinyVLM) -> 'TinyRewardModel':
        """Initialise the trunk from a copy of the SFT policy, as InstructGPT does."""
        raise NotImplementedError('TODO: implement from_policy (see the reference in src/mlbook)')

    def score(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``(B,)`` scalar reward per sequence."""
        raise NotImplementedError('TODO: implement score (see the reference in src/mlbook)')

def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """``-log sigma(r_w - r_l)``, averaged over the batch. Inputs ``(B,)``, output scalar.

    This is logistic regression on the reward difference, which is why the
    gradient is ``(sigma(r_w - r_l) - 1)`` times the difference of features: a
    pair the model already ranks confidently contributes almost nothing.
    """
    raise NotImplementedError('TODO: implement bradley_terry_loss (see the reference in src/mlbook)')

@dataclass
class PreferencePair:
    """One ``(x, y_w, y_l)`` triple over the same image and question."""
    example: task.Example
    chosen: str
    rejected: str

def _answer_space(answer: str) -> tuple[str, ...]:
    """The set of tokens a wrong answer can be drawn from, given the right one."""
    raise NotImplementedError('TODO: implement _answer_space (see the reference in src/mlbook)')

def make_preference_pairs(examples: list[task.Example], seed: int=0, policy: TinyVLM | None=None, temperature: float=1.0) -> list[PreferencePair]:
    """Build preferences with the verifier standing in for a human annotator.

    The chosen response is always the computed answer. The rejected response is
    either a random wrong token (``policy=None``) or a sample drawn from the
    policy and rejected by the verifier, which is the rejection-sampling recipe
    Llama 2 used to build its preference sets. On-policy negatives are the ones
    worth having: they are the mistakes the model actually makes, so the
    gradient lands where the errors are. Where the policy already answers
    correctly there is no negative to harvest, and a random wrong token stands
    in so that every prompt still contributes a pair.
    """
    raise NotImplementedError('TODO: implement make_preference_pairs (see the reference in src/mlbook)')

def _pair_batches(pairs: list[PreferencePair]) -> tuple:
    """Encode a list of pairs into a chosen batch and a rejected batch."""
    raise NotImplementedError('TODO: implement _pair_batches (see the reference in src/mlbook)')

def train_reward_model(rm: TinyRewardModel, pairs: list[PreferencePair], steps: int=150, batch_size: int=64, lr: float=0.001, seed: int=0) -> dict[str, object]:
    """Fit the scalar head (and the trunk) with the Bradley-Terry loss."""
    raise NotImplementedError('TODO: implement train_reward_model (see the reference in src/mlbook)')

@torch.no_grad()
def reward_pair_accuracy(rm: TinyRewardModel, pairs: list[PreferencePair], batch_size: int=128) -> float:
    """Fraction of held-out pairs the reward model ranks correctly (``r_w > r_l``)."""
    raise NotImplementedError('TODO: implement reward_pair_accuracy (see the reference in src/mlbook)')
