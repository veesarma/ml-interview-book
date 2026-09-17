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
    """Intra-op threads. Every tensor here is a few kilobytes, so the threading
    overhead of a parallel kernel costs far more than the kernel saves: on the
    machine this was developed on, four threads made one training step about
    250x slower than one thread. Set it to 0 to leave the global setting alone."""

    d_v: int = 32
    vision_heads: int = 2
    vision_layers: int = 2
    d_llm: int = 48
    lm_heads: int = 4
    lm_layers: int = 2
    n_query_tokens: int = 8

    sft_steps: int = 400
    sft_batch_size: int = 48
    sft_lr: float = 3e-3
    tool_fraction: float = 0.35

    rm_steps: int = 80
    rm_batch_size: int = 32
    rm_lr: float = 1e-3

    dpo_steps: int = 80
    dpo_batch_size: int = 24
    dpo_lr: float = 2e-4
    dpo_beta: float = 0.1

    grpo_steps: int = 25
    grpo_prompts_per_step: int = 8
    grpo_group_size: int = 6
    grpo_lr: float = 1e-4
    grpo_temperature: float = 1.0
    grpo_kl_coef: float = 0.02

    tool_confidence: float = 0.9


def build_model(config: PipelineConfig) -> TinyVLM:
    """Construct the VLM at the configured widths, on the configured seed."""
    torch.manual_seed(config.seed)
    return TinyVLM(
        vocab_size=task.VOCAB_SIZE,
        image_size=task.IMAGE_SIZE,
        patch_size=task.CELL,
        in_channels=3,
        d_v=config.d_v,
        vision_heads=config.vision_heads,
        vision_layers=config.vision_layers,
        d_llm=config.d_llm,
        lm_heads=config.lm_heads,
        lm_layers=config.lm_layers,
        n_query_tokens=config.n_query_tokens,
        max_text_len=task.MAX_TEXT_LEN,
    )


def majority_baseline(train: list[task.Example], evaluation: list[task.Example]) -> float:
    """Accuracy of always answering the most frequent training answer.

    Any stage that does not beat this has learned nothing about the images, and
    printing it next to the model's accuracy is the cheapest sanity check there
    is.
    """
    counts: dict[str, int] = {}
    for ex in train:
        counts[ex.answer] = counts.get(ex.answer, 0) + 1
    best = max(counts, key=lambda a: counts[a])
    return sum(1 for ex in evaluation if ex.answer == best) / max(len(evaluation), 1)


def run_pipeline(config: PipelineConfig | None = None, verbose: bool = False) -> dict[str, object]:
    """Run every stage end to end and return a report dict.

    Report keys: ``accuracy`` (per stage), ``sft``, ``reward_model``, ``dpo``,
    ``grpo``, ``tool_loop``, ``params``, ``tokens``, ``wall_time_s``, ``config``.
    """
    config = PipelineConfig() if config is None else config
    previous_threads = torch.get_num_threads()
    if config.torch_threads:
        torch.set_num_threads(config.torch_threads)
    try:
        return _run_pipeline(config, verbose)
    finally:
        torch.set_num_threads(previous_threads)


def _run_pipeline(config: PipelineConfig, verbose: bool) -> dict[str, object]:
    """The body of :func:`run_pipeline`, with the thread setting already applied."""
    torch.manual_seed(config.seed)
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    train = task.make_dataset(config.n_train, seed=config.seed)
    evaluation = task.make_dataset(config.n_eval, seed=config.seed + 10_000)
    timings["data"] = time.perf_counter() - t0

    model = build_model(config)
    acc: dict[str, float] = {"majority_baseline": majority_baseline(train, evaluation)}
    acc["random_init"] = accuracy(model, evaluation)

    t0 = time.perf_counter()
    sft_report = run_sft(
        model, train,
        steps=config.sft_steps, batch_size=config.sft_batch_size,
        lr=config.sft_lr, tool_fraction=config.tool_fraction, seed=config.seed,
    )
    timings["sft"] = time.perf_counter() - t0
    acc["sft"] = accuracy(model, evaluation)

    t0 = time.perf_counter()
    train_pairs = make_preference_pairs(train, seed=config.seed)
    eval_pairs = make_preference_pairs(evaluation, seed=config.seed + 1)
    rm = TinyRewardModel.from_policy(model)
    rm_report = train_reward_model(
        rm, train_pairs, steps=config.rm_steps,
        batch_size=config.rm_batch_size, lr=config.rm_lr, seed=config.seed,
    )
    rm_report["pair_accuracy"] = reward_pair_accuracy(rm, eval_pairs)
    timings["reward_model"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dpo_reference = freeze_reference(model)
    dpo_report = run_dpo(
        model, dpo_reference, train_pairs,
        steps=config.dpo_steps, batch_size=config.dpo_batch_size,
        lr=config.dpo_lr, beta=config.dpo_beta, seed=config.seed,
    )
    timings["dpo"] = time.perf_counter() - t0
    acc["dpo"] = accuracy(model, evaluation)

    t0 = time.perf_counter()
    grpo_reference = freeze_reference(model)
    grpo_report = run_grpo(
        model, grpo_reference, train,
        steps=config.grpo_steps, prompts_per_step=config.grpo_prompts_per_step,
        group_size=config.grpo_group_size, lr=config.grpo_lr,
        kl_coef=config.grpo_kl_coef, temperature=config.grpo_temperature, seed=config.seed,
    )
    timings["grpo"] = time.perf_counter() - t0
    acc["grpo"] = accuracy(model, evaluation)

    t0 = time.perf_counter()
    trajectories = run_episodes(model, evaluation, confidence_threshold=config.tool_confidence)
    tool_report = trajectory_report(trajectories)
    timings["tool_loop"] = time.perf_counter() - t0
    acc["tool_loop"] = tool_report["accuracy"]

    timings["total"] = time.perf_counter() - t_start

    report: dict[str, object] = {
        "accuracy": acc,
        "sft": {k: v for k, v in sft_report.items() if k != "losses"},
        "reward_model": {k: v for k, v in rm_report.items() if k != "losses"},
        "dpo": {k: v for k, v in dpo_report.items() if k not in ("losses", "margins")},
        "grpo": {k: v for k, v in grpo_report.items() if k not in ("losses", "mean_rewards")},
        "tool_loop": tool_report,
        "params": {
            "total": count_parameters(model),
            "vision": count_parameters(model.vision),
            "projector": count_parameters(model.projector),
            "lm": count_parameters(model.lm),
        },
        "tokens": {
            "vocab_size": task.VOCAB_SIZE,
            "visual_tokens_per_image": model.n_query_tokens,
            "patches_per_image": model.vision.n_tokens,
            "supervised_tokens": sft_report["supervised_tokens"],
        },
        "wall_time_s": {k: round(v, 3) for k, v in timings.items()},
        "config": asdict(config),
    }
    if verbose:
        print(format_report(report))
    return report


def format_report(report: dict[str, object]) -> str:
    """Pretty-print the accuracy column and the timings."""
    acc = report["accuracy"]                                             # type: ignore[index]
    lines = ["stage             accuracy"]
    for name in ("majority_baseline", "random_init", "sft", "dpo", "grpo", "tool_loop"):
        lines.append(f"{name:<18}{acc[name]:.3f}")                       # type: ignore[index]
    lines.append("")
    lines.append(f"parameters: {report['params']['total']:,}")           # type: ignore[index]
    lines.append(f"wall time:  {report['wall_time_s']['total']:.1f}s")   # type: ignore[index]
    return "\n".join(lines)


if __name__ == "__main__":
    run_pipeline(verbose=True)
