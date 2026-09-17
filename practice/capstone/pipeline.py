# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/pipeline.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k pipeline -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/pipeline --force

"""Canon #60, part 9 -- the whole pipeline in one function.

``run_pipeline()`` builds the data, builds the model, and runs

    SFT  ->  reward model  ->  DPO  ->  GRPO (verifiable reward)  ->  agent loop

measuring task accuracy after every stage. There is no pretraining: SFT starts
from random weights, which is the one compression this capstone makes so the
whole thing finishes in a few seconds on a CPU.

The report it returns is the artefact. Run it before and after any change you
make to a stage and read the accuracy column: a post-training stage that does
not move the number is either misconfigured or unnecessary, and being able to
say which is most of what a post-training job is.
"""
from __future__ import annotations
import time
from dataclasses import asdict, dataclass
import torch
from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM, count_parameters
from mlbook.capstone.preference_stage import freeze_reference, run_dpo, run_grpo
from mlbook.capstone.reward_stage import TinyRewardModel, make_preference_pairs, reward_pair_accuracy, train_reward_model
from mlbook.capstone.sft_stage import accuracy, run_sft
from mlbook.capstone.tool_loop import run_episodes, trajectory_report

@dataclass
class PipelineConfig:
    """Every knob, in one place, with the defaults that finish in seconds."""
    seed: int = 0
    n_train: int = 768
    n_eval: int = 192
    torch_threads: int = 1
    'Intra-op threads. Every tensor here is a few kilobytes, so the threading\n    overhead of a parallel kernel costs far more than the kernel saves: on the\n    machine this was developed on, four threads made one training step about\n    250x slower than one thread. Set it to 0 to leave the global setting alone.'
    d_v: int = 32
    vision_heads: int = 2
    vision_layers: int = 2
    d_llm: int = 48
    lm_heads: int = 4
    lm_layers: int = 2
    n_query_tokens: int = 8
    sft_steps: int = 400
    sft_batch_size: int = 48
    sft_lr: float = 0.003
    tool_fraction: float = 0.35
    rm_steps: int = 80
    rm_batch_size: int = 32
    rm_lr: float = 0.001
    dpo_steps: int = 80
    dpo_batch_size: int = 24
    dpo_lr: float = 0.0002
    dpo_beta: float = 0.1
    grpo_steps: int = 30
    grpo_prompts_per_step: int = 6
    grpo_group_size: int = 8
    grpo_oversample: int = 4
    'Prompts drawn per step, as a multiple of ``grpo_prompts_per_step``. Only\n    groups whose samples did not all score the same survive into the batch, so\n    the extra draws are what keep the batch full.'
    grpo_lr: float = 0.0003
    grpo_temperature: float = 0.7
    'Sampling temperature for the group. GRPO maximises the expected reward of\n    a *sample*, so a temperature far from the one you decode at optimises a\n    distribution you never serve: at 1.0 the mean accuracy change over five\n    seeds was -0.004, at 0.7 it was +0.019.'
    grpo_kl_coef: float = 0.02
    tool_confidence: float = 0.9

def build_model(config: PipelineConfig) -> TinyVLM:
    """Construct the VLM at the configured widths, on the configured seed."""
    raise NotImplementedError('TODO: implement build_model (see the reference in src/mlbook)')

def majority_baseline(train: list[task.Example], evaluation: list[task.Example]) -> float:
    """Accuracy of always answering the most frequent training answer.

    Any stage that does not beat this has learned nothing about the images, and
    printing it next to the model's accuracy is the cheapest sanity check there
    is.
    """
    raise NotImplementedError('TODO: implement majority_baseline (see the reference in src/mlbook)')

def run_pipeline(config: PipelineConfig | None=None, verbose: bool=False) -> dict[str, object]:
    """Run every stage end to end and return a report dict.

    Report keys: ``accuracy`` (per stage), ``sft``, ``reward_model``, ``dpo``,
    ``grpo``, ``tool_loop``, ``params``, ``tokens``, ``wall_time_s``, ``config``.
    """
    raise NotImplementedError('TODO: implement run_pipeline (see the reference in src/mlbook)')

def _run_pipeline(config: PipelineConfig, verbose: bool) -> dict[str, object]:
    """The body of :func:`run_pipeline`, with the thread setting already applied."""
    raise NotImplementedError('TODO: implement _run_pipeline (see the reference in src/mlbook)')

def format_report(report: dict[str, object]) -> str:
    """Pretty-print the accuracy column and the timings."""
    raise NotImplementedError('TODO: implement format_report (see the reference in src/mlbook)')
