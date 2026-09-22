"""Canon #60 end-to-end test: the whole pipeline in one call.

The configuration here is smaller than :class:`PipelineConfig`'s defaults so the
test stays inside the repository's ten-second-per-test budget. The default
configuration is the one the chapter reports, and it finishes in about twenty
seconds on one core.
"""

from __future__ import annotations

import math

import pytest
import torch

from mlbook.capstone import synthetic_task as task
from mlbook.capstone.pipeline import PipelineConfig, build_model, format_report, majority_baseline, run_pipeline


@pytest.fixture(autouse=True, scope="module")
def _single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def smoke_config() -> PipelineConfig:
    """A smaller run: the same code path with a shorter SFT stage.

    The GRPO settings are left at their defaults because that stage's behaviour
    depends on them (see :func:`mlbook.capstone.preference_stage.run_grpo`), and
    a shortened GRPO run stops part way through its drift rather than converging.
    """
    return PipelineConfig(
        seed=0,
        n_train=480,
        n_eval=192,
        sft_steps=300,
        sft_batch_size=32,
        rm_steps=60,
        rm_batch_size=24,
        dpo_steps=40,
        dpo_batch_size=16,
    )


@pytest.fixture(scope="module")
def report() -> dict:
    return run_pipeline(smoke_config())


def test_report_has_every_stage(report):
    for key in ("accuracy", "sft", "reward_model", "dpo", "grpo", "tool_loop", "params", "tokens", "wall_time_s"):
        assert key in report
    for stage in ("majority_baseline", "random_init", "sft", "dpo", "grpo", "tool_loop"):
        assert 0.0 <= report["accuracy"][stage] <= 1.0
    assert report["wall_time_s"]["total"] > 0.0
    assert set(report["wall_time_s"]) >= {"data", "sft", "reward_model", "dpo", "grpo", "tool_loop", "total"}
    assert "accuracy" in format_report(report)


def test_the_model_is_tiny_and_the_token_budget_is_reported(report):
    assert report["params"]["total"] < 200_000
    assert report["params"]["total"] == (
        report["params"]["vision"] + report["params"]["projector"] + report["params"]["lm"]
    )
    assert report["tokens"]["vocab_size"] == task.VOCAB_SIZE
    assert report["tokens"]["patches_per_image"] == 16
    assert report["tokens"]["visual_tokens_per_image"] == 8
    assert report["tokens"]["supervised_tokens"] == 300 * 32 * 2


def test_sft_learns_the_task(report):
    acc = report["accuracy"]
    assert report["sft"]["loss_last"] < report["sft"]["loss_first"]
    assert acc["sft"] > acc["random_init"]
    assert acc["sft"] > acc["majority_baseline"], "SFT must beat answering the modal token"


def test_reward_model_ranks_preferences(report):
    rm = report["reward_model"]
    assert rm["loss_last"] < rm["loss_first"]
    assert rm["pair_accuracy"] > 0.5, "a Bradley-Terry head must beat a coin flip"


def test_post_training_does_not_reduce_accuracy(report):
    """Neither preference stage may fall materially below the SFT policy.

    The bound is two standard errors of a binomial proportion at the SFT accuracy,
    about 0.07 at this ``n_eval``. The previous fixed bound of 0.02 was tighter than
    the measurement itself: 192 evaluation examples put one standard error near an
    accuracy of 0.55 at 0.036, so 0.02 asked the test to resolve a difference half
    the size of its own noise. It duly failed on CI, where a different torch build
    draws a different RNG stream and DPO landed 0.031 under SFT, which is inside
    one standard error and therefore not evidence of a regression.

    Widening the band alone would weaken the test, so the floor against the majority
    baseline is asserted too. Together they still catch what this test exists for: a
    preference stage that collapses toward the 0.22 baseline is a drop of over 0.3,
    which is more than eight standard errors.
    """
    acc = report["accuracy"]
    p = acc["sft"]
    n_eval = smoke_config().n_eval
    tolerance = 2.0 * math.sqrt(p * (1.0 - p) / n_eval)  # ~0.07 at n_eval = 192
    assert acc["dpo"] >= acc["sft"] - tolerance
    assert acc["grpo"] >= acc["sft"] - tolerance
    assert acc["dpo"] > acc["majority_baseline"]
    assert acc["grpo"] > acc["majority_baseline"]
    assert report["dpo"]["margin_before"] == pytest.approx(0.0, abs=1e-5)
    assert report["dpo"]["margin_after"] > report["dpo"]["margin_before"]


def test_grpo_optimises_the_verifier_reward(report):
    """GRPO's objective is the expected reward of a *sample*, so that is what is
    checked here; greedy exact match is a different statistic and moves later."""
    grpo = report["grpo"]
    assert 0.0 <= grpo["reward_first"] <= 1.0
    assert 0.0 <= grpo["reward_last"] <= 1.0
    assert grpo["reward_last"] >= grpo["reward_first"] - 0.05


def test_grpo_drops_the_degenerate_groups(report):
    """Dynamic sampling has to be doing something and not everything: a rate of
    1.0 would mean the filter never fired, a rate of 0.0 that nothing survived."""
    rate = report["grpo"]["informative_group_rate"]
    assert 0.1 < rate < 0.95
    assert report["grpo"]["groups_used"] > 0


def test_tool_loop_helps(report):
    tool = report["tool_loop"]
    assert tool["well_formed_rate"] == 1.0
    assert 1.0 <= tool["mean_steps"] <= 2.0
    assert tool["accuracy"] >= report["accuracy"]["grpo"]


def test_majority_baseline_is_computed_from_the_training_answers():
    train = task.make_dataset(64, seed=0)
    assert 0.0 < majority_baseline(train, train) < 1.0
    assert majority_baseline(train, []) == 0.0


def test_build_model_is_reproducible():
    config = PipelineConfig(seed=11)
    a = build_model(config)
    b = build_model(config)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
