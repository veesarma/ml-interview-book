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

# ---------------------------------------------------------------------------
# The verifiable reward
# ---------------------------------------------------------------------------


def verifier_reward(prediction: str, example: task.Example) -> float:
    """Programmatic reward for one answer. 1.0 if it matches the computed answer, else 0.0.

    The gold answer comes from :func:`mlbook.capstone.synthetic_task.count_shapes`
    or :func:`~mlbook.capstone.synthetic_task.largest_colour` run on the scene
    graph that drew the image, so no annotation and no judge model is involved.
    """
    return 1.0 if prediction == example.answer else 0.0


def verifier_rewards(predictions: list[str], examples: list[task.Example]) -> torch.Tensor:
    """Vectorised :func:`verifier_reward`. Returns ``(B,)`` float32."""
    values = [verifier_reward(p, ex) for p, ex in zip(predictions, examples)]
    return torch.tensor(values, dtype=torch.float32)                     # (B,)


# ---------------------------------------------------------------------------
# The learned reward model
# ---------------------------------------------------------------------------


class TinyRewardModel(nn.Module):
    """A VLM trunk with a scalar head instead of a vocabulary head.

    ``score(images, input_ids, attention_mask)`` returns ``(B,)``: the value of
    the final hidden state at the last real token of the sequence, which is the
    standard reward-model head (read the end of the response, score the whole
    thing).
    """

    def __init__(self, trunk: TinyVLM) -> None:
        super().__init__()
        self.trunk = trunk
        self.value_head = nn.Linear(trunk.lm.d_model, 1)
        nn.init.zeros_(self.value_head.bias)
        nn.init.normal_(self.value_head.weight, std=0.02)

    @classmethod
    def from_policy(cls, policy: TinyVLM) -> "TinyRewardModel":
        """Initialise the trunk from a copy of the SFT policy, as InstructGPT does."""
        return cls(copy.deepcopy(policy))

    def score(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(B,)`` scalar reward per sequence."""
        h = self.trunk.hidden_states(images, input_ids, attention_mask)  # (B, S, d)
        n_real = attention_mask.sum(dim=1)                               # (B,)
        last = n_real + self.trunk.n_query_tokens - 1                    # (B,) absolute index
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)  # (B,)
        pooled = h[rows, last, :]                                        # (B, d)
        return self.value_head(pooled)[:, 0]                             # (B,)


def bradley_terry_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """``-log sigma(r_w - r_l)``, averaged over the batch. Inputs ``(B,)``, output scalar.

    This is logistic regression on the reward difference, which is why the
    gradient is ``(sigma(r_w - r_l) - 1)`` times the difference of features: a
    pair the model already ranks confidently contributes almost nothing.
    """
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


# ---------------------------------------------------------------------------
# Preference data
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """One ``(x, y_w, y_l)`` triple over the same image and question."""

    example: task.Example
    chosen: str
    rejected: str


def _answer_space(answer: str) -> tuple[str, ...]:
    """The set of tokens a wrong answer can be drawn from, given the right one."""
    return task.COLOURS if answer in task.COLOURS else tuple(str(i) for i in range(task.MAX_SHAPES + 1))


def make_preference_pairs(
    examples: list[task.Example],
    seed: int = 0,
    policy: TinyVLM | None = None,
    temperature: float = 1.0,
) -> list[PreferencePair]:
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
    rng = np.random.default_rng(seed)
    sampled: list[str] | None = None
    if policy is not None:
        was_training = policy.training
        policy.eval()
        with torch.no_grad():
            logits = next_token_logits(policy, examples)                 # (B, V)
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)   # (B, V)
        generator = torch.Generator().manual_seed(seed)
        draws = torch.multinomial(probs, num_samples=1, generator=generator)[:, 0]  # (B,)
        sampled = [task.ITOS[int(i)] for i in draws]
        policy.train(was_training)

    pairs: list[PreferencePair] = []
    for i, ex in enumerate(examples):
        space = [a for a in _answer_space(ex.answer) if a != ex.answer]
        rejected = space[int(rng.integers(len(space)))]
        if sampled is not None and sampled[i] in space:
            rejected = sampled[i]
        pairs.append(PreferencePair(example=ex, chosen=ex.answer, rejected=rejected))
    return pairs


def _pair_batches(pairs: list[PreferencePair]) -> tuple:
    """Encode a list of pairs into a chosen batch and a rejected batch."""
    examples = [p.example for p in pairs]
    chosen = encode_batch(examples, answers=[p.chosen for p in pairs])
    rejected = encode_batch(examples, answers=[p.rejected for p in pairs])
    return chosen, rejected


def train_reward_model(
    rm: TinyRewardModel,
    pairs: list[PreferencePair],
    steps: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, object]:
    """Fit the scalar head (and the trunk) with the Bradley-Terry loss."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    rm.train()
    for _ in range(steps):
        idx = rng.integers(0, len(pairs), size=batch_size)
        chunk = [pairs[int(i)] for i in idx]
        chosen, rejected = _pair_batches(chunk)
        r_w = rm.score(chosen.images, chosen.input_ids, chosen.attention_mask)     # (B,)
        r_l = rm.score(rejected.images, rejected.input_ids, rejected.attention_mask)  # (B,)
        loss = bradley_terry_loss(r_w, r_l)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "losses": losses,
        "loss_first": float(np.mean(losses[:10])),
        "loss_last": float(np.mean(losses[-10:])),
    }


@torch.no_grad()
def reward_pair_accuracy(rm: TinyRewardModel, pairs: list[PreferencePair], batch_size: int = 128) -> float:
    """Fraction of held-out pairs the reward model ranks correctly (``r_w > r_l``)."""
    rm.eval()
    correct = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        chosen, rejected = _pair_batches(chunk)
        r_w = rm.score(chosen.images, chosen.input_ids, chosen.attention_mask)     # (b,)
        r_l = rm.score(rejected.images, rejected.input_ids, rejected.attention_mask)  # (b,)
        correct += int((r_w > r_l).sum())
    return correct / max(len(pairs), 1)
