# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/sft_stage.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k sft_stage -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/sft_stage --force

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
    images: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    prompt_len: torch.Tensor

def _pad_rows(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad ragged token lists. Returns ``(ids (B, T), mask (B, T))``."""
    raise NotImplementedError('TODO: implement _pad_rows (see the reference in src/mlbook)')

def encode_batch(examples: list[task.Example], observations: list[str | None] | None=None, answers: list[str] | None=None, include_eos: bool=True, device: torch.device | str='cpu') -> SFTBatch:
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
    raise NotImplementedError('TODO: implement encode_batch (see the reference in src/mlbook)')

def sft_loss(model: TinyVLM, batch: SFTBatch) -> torch.Tensor:
    """Mean cross-entropy over assistant tokens. Scalar.

    ``text_logits[:, t]`` already scores text token ``t``, so the masked gather
    needs no further shifting.
    """
    raise NotImplementedError('TODO: implement sft_loss (see the reference in src/mlbook)')

def next_token_logits(model: TinyVLM, examples: list[task.Example], observations: list[str | None] | None=None) -> torch.Tensor:
    """Logits for the first assistant token of each prompt. Returns ``(B, V)``.

    Prompts have different lengths (four words for "how many shapes", six for
    "what colour is the largest shape"), so after right-padding the prediction
    slot is a different column in every row. Gathering at
    ``N_q + prompt_len - 1`` picks the right one; taking ``logits[:, -1]`` would
    read a padding position for the short rows.
    """
    raise NotImplementedError('TODO: implement next_token_logits (see the reference in src/mlbook)')

@torch.no_grad()
def predict_answers(model: TinyVLM, examples: list[task.Example], observations: list[str | None] | None=None, batch_size: int=256) -> list[str]:
    """Greedy answer for each example, decoded to a vocabulary token.

    Every answer in this task is one token, so greedy decoding is one argmax at
    the first assistant position.
    """
    raise NotImplementedError('TODO: implement predict_answers (see the reference in src/mlbook)')

def accuracy(model: TinyVLM, examples: list[task.Example], observations: list[str | None] | None=None) -> float:
    """Fraction of examples whose greedy answer equals the computed answer."""
    raise NotImplementedError('TODO: implement accuracy (see the reference in src/mlbook)')

def choose_observations(examples: list[task.Example], tool_fraction: float, rng: np.random.Generator) -> list[str | None]:
    """Decide which training rows carry a tool observation.

    A fraction of rows become ``question observation : <n> answer :`` with the
    correct count spliced in, which is what teaches the model to read an
    observation. Without those rows the agent loop in
    :mod:`mlbook.capstone.tool_loop` changes nothing, because a model that has
    never seen an observation token has no reason to condition on one.
    """
    raise NotImplementedError('TODO: implement choose_observations (see the reference in src/mlbook)')

def run_sft(model: TinyVLM, examples: list[task.Example], steps: int=600, batch_size: int=64, lr: float=0.003, tool_fraction: float=0.35, seed: int=0) -> dict[str, object]:
    """Train vision, projector and LM jointly on the assistant-only loss.

    Returns the loss curve, the mean of its first and last ten steps, and the
    number of supervised tokens seen.
    """
    raise NotImplementedError('TODO: implement run_sft (see the reference in src/mlbook)')
