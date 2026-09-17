"""Canon #60, part 5 -- supervised fine-tuning with an assistant-only loss.

The model has never seen a token of text before this stage: there is no
pretraining corpus, so SFT does all the work of teaching the format and the
task. That is the one place this capstone departs from a real pipeline, and it
is deliberate, because it keeps the whole run under a minute on a laptop.

The loss detail interviewers ask about is the mask. Tokens in the prompt (the
question, the optional tool observation, and the ``answer :`` marker) are
conditioning, not targets. Training on them spends capacity modelling the
question distribution and lets the model score well by memorising prompts.
``loss_mask`` marks the assistant turn and nothing else.

This module also owns the batching helpers, because every later stage (reward
model, DPO, GRPO, the agent loop) needs the same padded tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM


@dataclass
class SFTBatch:
    """One padded batch. ``B`` examples, ``T`` text tokens."""

    images: torch.Tensor          # (B, 3, 16, 16) float32
    input_ids: torch.Tensor       # (B, T) long, right-padded with <pad>
    attention_mask: torch.Tensor  # (B, T) long, 1 on real tokens
    loss_mask: torch.Tensor       # (B, T) bool, 1 on assistant tokens only
    prompt_len: torch.Tensor      # (B,) long, number of prompt tokens per row


def _pad_rows(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad ragged token lists. Returns ``(ids (B, T), mask (B, T))``."""
    B = len(rows)
    T = max(len(r) for r in rows)
    ids = np.full((B, T), task.PAD_ID, dtype=np.int64)                   # (B, T)
    mask = np.zeros((B, T), dtype=np.int64)                              # (B, T)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = row
        mask[i, :len(row)] = 1
    return ids, mask


def encode_batch(
    examples: list[task.Example],
    observations: list[str | None] | None = None,
    answers: list[str] | None = None,
    include_eos: bool = True,
    device: torch.device | str = "cpu",
) -> SFTBatch:
    """Tokenise and pad examples into an :class:`SFTBatch`.

    Args:
        examples: the (image, question, answer) triples.
        observations: tool results to splice into each prompt; ``None`` entries
            give a plain prompt.
        answers: assistant turns to supervise, overriding the gold answers. DPO
            and GRPO pass rejected or sampled answers here.
        include_eos: append ``<eos>`` to the assistant turn. GRPO turns it off
            so a sampled response is exactly one token long.
    """
    observations = [None] * len(examples) if observations is None else observations
    answers = [ex.answer for ex in examples] if answers is None else answers
    prompts = [task.build_prompt(ex.question, obs) for ex, obs in zip(examples, observations)]
    targets = [task.encode(a) + ([task.EOS_ID] if include_eos else []) for a in answers]

    ids, mask = _pad_rows([p + t for p, t in zip(prompts, targets)])      # (B, T), (B, T)
    loss_mask = np.zeros_like(mask, dtype=bool)                          # (B, T)
    for i, (prompt, target) in enumerate(zip(prompts, targets)):
        loss_mask[i, len(prompt):len(prompt) + len(target)] = True       # assistant turn only
    images = np.stack([ex.image for ex in examples])                     # (B, 3, 16, 16)
    return SFTBatch(
        images=torch.as_tensor(images, dtype=torch.float32, device=device),
        input_ids=torch.as_tensor(ids, device=device),
        attention_mask=torch.as_tensor(mask, device=device),
        loss_mask=torch.as_tensor(loss_mask, device=device),
        prompt_len=torch.as_tensor([len(p) for p in prompts], device=device),
    )


def sft_loss(model: TinyVLM, batch: SFTBatch) -> torch.Tensor:
    """Mean cross-entropy over assistant tokens. Scalar.

    ``text_logits[:, t]`` already scores text token ``t``, so the masked gather
    needs no further shifting.
    """
    logits = model.text_logits(batch.images, batch.input_ids, batch.attention_mask)  # (B, T, V)
    selected = logits[batch.loss_mask]                                   # (n_assistant, V)
    targets = batch.input_ids[batch.loss_mask]                           # (n_assistant,)
    return F.cross_entropy(selected, targets)


def next_token_logits(
    model: TinyVLM,
    examples: list[task.Example],
    observations: list[str | None] | None = None,
) -> torch.Tensor:
    """Logits for the first assistant token of each prompt. Returns ``(B, V)``.

    Prompts have different lengths (four words for "how many shapes", six for
    "what colour is the largest shape"), so after right-padding the prediction
    slot is a different column in every row. Gathering at
    ``N_q + prompt_len - 1`` picks the right one; taking ``logits[:, -1]`` would
    read a padding position for the short rows.
    """
    observations = [None] * len(examples) if observations is None else observations
    prompts = [task.build_prompt(ex.question, obs) for ex, obs in zip(examples, observations)]
    ids, mask = _pad_rows(prompts)                                       # (B, T_p), (B, T_p)
    images = torch.as_tensor(np.stack([ex.image for ex in examples]), dtype=torch.float32)  # (B, 3, 16, 16)
    logits = model(images, torch.as_tensor(ids), torch.as_tensor(mask))  # (B, N_q + T_p, V)
    last = torch.as_tensor([len(p) for p in prompts]) + model.n_query_tokens - 1  # (B,)
    rows = torch.arange(len(examples))                                   # (B,)
    return logits[rows, last, :]                                         # (B, V)


@torch.no_grad()
def predict_answers(
    model: TinyVLM,
    examples: list[task.Example],
    observations: list[str | None] | None = None,
    batch_size: int = 256,
) -> list[str]:
    """Greedy answer for each example, decoded to a vocabulary token.

    Every answer in this task is one token, so greedy decoding is one argmax at
    the first assistant position.
    """
    was_training = model.training
    model.eval()
    out: list[str] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        obs = None if observations is None else observations[start:start + batch_size]
        logits = next_token_logits(model, chunk, obs)                    # (b, V)
        out.extend(task.ITOS[int(i)] for i in logits.argmax(dim=-1))
    model.train(was_training)
    return out


def accuracy(
    model: TinyVLM,
    examples: list[task.Example],
    observations: list[str | None] | None = None,
) -> float:
    """Fraction of examples whose greedy answer equals the computed answer."""
    preds = predict_answers(model, examples, observations)
    return float(np.mean([p == ex.answer for p, ex in zip(preds, examples)]))


def choose_observations(
    examples: list[task.Example],
    tool_fraction: float,
    rng: np.random.Generator,
) -> list[str | None]:
    """Decide which training rows carry a tool observation.

    A fraction of rows become ``question observation : <n> answer :`` with the
    correct count spliced in, which is what teaches the model to read an
    observation. Without those rows the agent loop in
    :mod:`mlbook.capstone.tool_loop` changes nothing, because a model that has
    never seen an observation token has no reason to condition on one.
    """
    observations: list[str | None] = []
    for ex in examples:
        if ex.tool_query is not None and rng.random() < tool_fraction:
            colour, kind = ex.tool_query
            observations.append(str(task.count_shapes(ex.scene, colour, kind)))
        else:
            observations.append(None)
    return observations


def run_sft(
    model: TinyVLM,
    examples: list[task.Example],
    steps: int = 600,
    batch_size: int = 64,
    lr: float = 3e-3,
    tool_fraction: float = 0.35,
    seed: int = 0,
) -> dict[str, object]:
    """Train vision, projector and LM jointly on the assistant-only loss.

    Returns the loss curve, the mean of its first and last ten steps, and the
    number of supervised tokens seen.
    """
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses: list[float] = []
    supervised_tokens = 0
    model.train()
    for _ in range(steps):
        idx = rng.integers(0, len(examples), size=batch_size)
        chunk = [examples[int(i)] for i in idx]
        batch = encode_batch(chunk, choose_observations(chunk, tool_fraction, rng))
        loss = sft_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        supervised_tokens += int(batch.loss_mask.sum())
    return {
        "losses": losses,
        "loss_first": float(np.mean(losses[:10])),
        "loss_last": float(np.mean(losses[-10:])),
        "supervised_tokens": supervised_tokens,
    }
