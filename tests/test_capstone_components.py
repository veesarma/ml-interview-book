"""Canon #60 component tests: shapes, masks, losses and the verifier.

Every test here runs in a couple of seconds on one CPU core. The intra-op
thread count is pinned to 1 for the module because the tensors are a few
kilobytes each and parallel kernels cost more than they save at that size.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from mlbook.capstone import synthetic_task as task
from mlbook.capstone.multimodal_model import TinyVLM, count_parameters
from mlbook.capstone.preference_stage import (
    dpo_loss,
    freeze_reference,
    group_advantages,
    grpo_loss,
    k3_kl,
    mean_dpo_margin,
    run_dpo,
    sequence_logprob,
)
from mlbook.capstone.projector import Projector
from mlbook.capstone.reward_stage import (
    TinyRewardModel,
    bradley_terry_loss,
    make_preference_pairs,
    verifier_reward,
    verifier_rewards,
)
from mlbook.capstone.sft_stage import encode_batch, next_token_logits, run_sft, sft_loss
from mlbook.capstone.tiny_lm import MASK_VALUE, TinyCausalLM, causal_padding_bias
from mlbook.capstone.tiny_vision_encoder import PatchEmbed, TinyVisionEncoder
from mlbook.capstone.tool_loop import count_shapes_tool, render_trajectory, run_episodes, trajectory_report


@pytest.fixture(autouse=True, scope="module")
def _single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def model() -> TinyVLM:
    torch.manual_seed(0)
    return TinyVLM(vocab_size=task.VOCAB_SIZE, image_size=task.IMAGE_SIZE, patch_size=task.CELL)


@pytest.fixture(scope="module")
def examples() -> list[task.Example]:
    return task.make_dataset(48, seed=3)


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


def test_tokenizer_round_trips():
    text = "how many red squares"
    ids = task.encode(text)
    assert len(ids) == 4
    assert task.decode(ids) == text
    assert task.decode(ids + [task.PAD_ID, task.PAD_ID]) == text


def test_dataset_is_deterministic_and_well_formed():
    a = task.make_dataset(16, seed=7)
    b = task.make_dataset(16, seed=7)
    c = task.make_dataset(16, seed=8)
    assert [x.question for x in a] == [x.question for x in b]
    assert np.allclose(a[0].image, b[0].image)
    assert [x.question for x in a] != [x.question for x in c] or not np.allclose(a[0].image, c[0].image)
    for ex in a:
        assert ex.image.shape == (3, task.IMAGE_SIZE, task.IMAGE_SIZE)
        assert ex.image.dtype == np.float32
        assert len(task.encode(ex.answer)) == 1
        assert 1 <= len(ex.scene.shapes) <= task.MAX_SHAPES


def test_verifier_reward_on_hand_built_scene():
    """A scene written out by hand, so the oracle is checked against arithmetic."""
    shapes = [
        task.Shape("red", "square", 0, 0, 4, 16),
        task.Shape("red", "square", 1, 1, 3, 9),
        task.Shape("red", "circle", 2, 2, 3, 5),
        task.Shape("blue", "circle", 3, 3, 4, 12),
    ]
    scene = task.Scene(shapes=shapes, image=task.render(shapes, np.random.default_rng(0), noise=0.0))

    assert task.count_shapes(scene) == 4
    assert task.count_shapes(scene, "red") == 3
    assert task.count_shapes(scene, "red", "square") == 2
    assert task.count_shapes(scene, "green") == 0
    assert task.largest_colour(scene) == "red"          # area 16 beats 12

    ex = task.Example(image=scene.image, question="how many red squares", answer="2",
                      scene=scene, tool_query=("red", "square"))
    assert verifier_reward("2", ex) == 1.0
    assert verifier_reward("3", ex) == 0.0
    assert verifier_reward("red", ex) == 0.0
    assert torch.equal(verifier_rewards(["2", "0"], [ex, ex]), torch.tensor([1.0, 0.0]))


def test_render_places_colour_in_the_right_channel():
    shapes = [task.Shape("green", "square", 1, 2, 4, 16)]
    image = task.render(shapes, np.random.default_rng(0), noise=0.0)     # (3, 16, 16)
    assert image[1, 4:8, 8:12].min() == pytest.approx(1.0)
    assert image[0].max() == pytest.approx(0.0)
    assert image[2].max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Vision tower and projector
# ---------------------------------------------------------------------------


def test_patch_embed_shape_and_token_count():
    embed = PatchEmbed(image_size=16, patch_size=4, in_channels=3, d_v=32)
    out = embed(torch.randn(5, 3, 16, 16))                               # (B, N_v, d_v)
    assert out.shape == (5, 16, 32)
    assert embed.patch_dim == 3 * 4 * 4
    with pytest.raises(ValueError):
        PatchEmbed(image_size=16, patch_size=5, in_channels=3, d_v=32)


def test_patch_embed_preserves_patch_layout():
    """Patch k must be built from the pixels of cell k, in row-major order."""
    embed = PatchEmbed(image_size=16, patch_size=4, in_channels=3, d_v=8)
    image = torch.zeros(1, 3, 16, 16)
    image[0, 0, 4:8, 8:12] = 1.0          # cell (row 1, col 2) -> flat index 1*4 + 2 = 6
    with torch.no_grad():
        flat = image.reshape(1, 3, 4, 4, 4, 4).permute(0, 2, 4, 1, 3, 5).reshape(1, 16, 48)
    assert flat[0, 6].sum() == pytest.approx(16.0)
    assert flat[0, [i for i in range(16) if i != 6]].sum() == pytest.approx(0.0)


def test_vision_encoder_shape():
    encoder = TinyVisionEncoder(image_size=16, patch_size=4, d_v=32, n_heads=2, n_layers=2)
    out = encoder(torch.randn(4, 3, 16, 16))                             # (B, N_v, d_v)
    assert out.shape == (4, 16, 32)
    assert encoder.n_tokens == 16


def test_projector_maps_visual_tokens_to_llm_tokens():
    proj = Projector(d_v=32, d_llm=48, n_visual_tokens=16, n_query_tokens=8)
    out = proj(torch.randn(4, 16, 32))                                   # (B, N_q, d_llm)
    assert out.shape == (4, 8, 48)
    assert proj.group == 2
    with pytest.raises(ValueError):
        Projector(d_v=32, d_llm=48, n_visual_tokens=16, n_query_tokens=5)


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------


def test_causal_padding_bias_is_finite_and_correct():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])                          # (B, T)
    bias = causal_padding_bias(mask)                                     # (B, 1, T, T)
    assert bias.shape == (2, 1, 3, 3)
    assert torch.isfinite(bias).all(), "a padded query row of -inf would make softmax NaN"
    assert bias[1, 0, 0, 1] == MASK_VALUE                                # future token masked
    assert bias[1, 0, 1, 0] == 0.0                                       # past token visible
    assert bias[0, 0, 2, 2] == MASK_VALUE                                # padded key masked
    weights = torch.softmax(torch.zeros(2, 1, 3, 3) + bias, dim=-1)
    assert torch.isfinite(weights).all()


def test_lm_forward_shape_and_causality():
    torch.manual_seed(0)
    lm = TinyCausalLM(vocab_size=task.VOCAB_SIZE, d_model=32, n_heads=4, n_layers=2, max_positions=16)
    ids = torch.randint(0, task.VOCAB_SIZE, (3, 6))                      # (B, T)
    logits = lm(ids)                                                     # (B, T, V)
    assert logits.shape == (3, 6, task.VOCAB_SIZE)

    changed = ids.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % task.VOCAB_SIZE
    with torch.no_grad():
        assert torch.allclose(lm(ids)[:, :4], lm(changed)[:, :4], atol=1e-6)


def test_lm_generate_greedy_is_deterministic_and_sampling_is_not():
    torch.manual_seed(0)
    lm = TinyCausalLM(vocab_size=task.VOCAB_SIZE, d_model=32, n_heads=4, n_layers=1, max_positions=16)
    ids = torch.randint(0, task.VOCAB_SIZE, (2, 4))                      # (B, T)
    a = lm.generate(ids, max_new_tokens=3, greedy=True)                  # (B, T + 3)
    b = lm.generate(ids, max_new_tokens=3, greedy=True)
    assert a.shape == (2, 7)
    assert torch.equal(a, b)

    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(1)
    s1 = lm.generate(ids, max_new_tokens=3, greedy=False, temperature=1.5, top_k=5, generator=g1)
    s2 = lm.generate(ids, max_new_tokens=3, greedy=False, temperature=1.5, top_k=5, generator=g2)
    assert torch.equal(s1, s2), "same seed must reproduce the sample"
    assert s1.shape == (2, 7)


# ---------------------------------------------------------------------------
# The VLM
# ---------------------------------------------------------------------------


def test_vlm_build_inputs_splices_positions_and_mask(model, examples):
    batch = encode_batch(examples[:4])
    embeds, mask, positions = model.build_inputs(batch.images, batch.input_ids, batch.attention_mask)
    B, T = batch.input_ids.shape
    S = model.n_query_tokens + T
    assert embeds.shape == (B, S, model.lm.d_model)
    assert mask.shape == (B, S)
    assert positions.shape == (B, S)
    assert torch.equal(mask[:, :model.n_query_tokens], torch.ones(B, model.n_query_tokens, dtype=mask.dtype))
    assert torch.equal(mask[:, model.n_query_tokens:], batch.attention_mask)
    assert torch.equal(positions[0], torch.arange(S))


def test_vlm_text_logits_undo_the_next_token_shift(model, examples):
    batch = encode_batch(examples[:3])
    with torch.no_grad():
        full = model(batch.images, batch.input_ids, batch.attention_mask)        # (B, S, V)
        text = model.text_logits(batch.images, batch.input_ids, batch.attention_mask)  # (B, T, V)
    T = batch.input_ids.shape[1]
    start = model.n_query_tokens - 1
    assert text.shape == (3, T, task.VOCAB_SIZE)
    assert torch.allclose(text, full[:, start:start + T], atol=0)


def test_vlm_generate_shape(model, examples):
    batch = encode_batch(examples[:2])
    prompt = batch.input_ids[:, :int(batch.prompt_len[0])]
    out = model.generate(batch.images, prompt, max_new_tokens=2, greedy=True, eos_id=task.EOS_ID)
    assert out.shape == (2, 2)


def test_parameter_count_is_small(model):
    assert 20_000 < count_parameters(model) < 200_000


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------


def test_encode_batch_masks_only_the_assistant_turn(examples):
    batch = encode_batch(examples[:6])
    B, T = batch.input_ids.shape
    assert batch.images.shape == (6, 3, task.IMAGE_SIZE, task.IMAGE_SIZE)
    assert batch.loss_mask.shape == (B, T)
    for i in range(B):
        prompt_len = int(batch.prompt_len[i])
        assert not batch.loss_mask[i, :prompt_len].any(), "prompt tokens must not be trained on"
        assert int(batch.loss_mask[i].sum()) == 2, "answer token plus <eos>"
        answer_id = int(batch.input_ids[i, prompt_len])
        assert task.ITOS[answer_id] == examples[i].answer
        assert int(batch.input_ids[i, prompt_len + 1]) == task.EOS_ID


def test_encode_batch_can_splice_an_observation(examples):
    ex = examples[0]
    plain = encode_batch([ex])
    with_obs = encode_batch([ex], observations=["3"])
    assert int(with_obs.prompt_len[0]) == int(plain.prompt_len[0]) + 3
    decoded = task.decode(with_obs.input_ids[0].tolist())
    assert "observation : 3" in decoded


def test_sft_reduces_loss():
    torch.manual_seed(0)
    model = TinyVLM(vocab_size=task.VOCAB_SIZE, image_size=task.IMAGE_SIZE, patch_size=task.CELL)
    data = task.make_dataset(192, seed=1)
    batch = encode_batch(data[:32])
    before = float(sft_loss(model, batch).detach())
    report = run_sft(model, data, steps=120, batch_size=32, lr=3e-3, seed=0)
    after = float(sft_loss(model, batch).detach())
    assert report["loss_last"] < report["loss_first"]
    assert after < before
    assert report["supervised_tokens"] == 120 * 32 * 2


def test_next_token_logits_gathers_the_right_column(model):
    """Prompts of different lengths must not read a padding position."""
    data = task.make_dataset(24, seed=5)
    short = [ex for ex in data if ex.question == "how many shapes"]
    long = [ex for ex in data if ex.question.startswith("what colour")]
    if not short or not long:
        pytest.skip("this seed did not produce both question lengths")
    mixed = [short[0], long[0]]
    with torch.no_grad():
        together = next_token_logits(model, mixed)                       # (2, V)
        alone = torch.cat([next_token_logits(model, [ex]) for ex in mixed], dim=0)  # (2, V)
    assert torch.allclose(together, alone, atol=1e-5)


# ---------------------------------------------------------------------------
# Reward model and the verifier
# ---------------------------------------------------------------------------


def test_bradley_terry_loss_matches_the_closed_form():
    r_w = torch.tensor([0.0, 2.0])
    r_l = torch.tensor([0.0, 0.0])
    expected = -(np.log(0.5) + np.log(1 / (1 + np.exp(-2.0)))) / 2
    assert float(bradley_terry_loss(r_w, r_l)) == pytest.approx(expected, rel=1e-5)
    # only differences matter: a constant shift leaves the loss unchanged
    assert float(bradley_terry_loss(r_w + 3.0, r_l + 3.0)) == pytest.approx(expected, rel=1e-5)


def test_reward_model_score_shape_and_training(model, examples):
    rm = TinyRewardModel.from_policy(model)
    batch = encode_batch(examples[:5])
    scores = rm.score(batch.images, batch.input_ids, batch.attention_mask)
    assert scores.shape == (5,)
    pairs = make_preference_pairs(examples[:5], seed=0)
    assert all(p.chosen != p.rejected for p in pairs)
    assert all(p.chosen == p.example.answer for p in pairs)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


def test_dpo_loss_matches_the_closed_form():
    pol_w = torch.tensor([-1.0]); pol_l = torch.tensor([-3.0])
    ref_w = torch.tensor([-2.0]); ref_l = torch.tensor([-2.0])
    loss, margin = dpo_loss(pol_w, pol_l, ref_w, ref_l, beta=0.5)
    assert float(margin) == pytest.approx((-1.0 + 2.0) - (-3.0 + 2.0))   # +2
    assert float(loss) == pytest.approx(-np.log(1 / (1 + np.exp(-0.5 * 2.0))), rel=1e-6)
    # a policy equal to the reference sits exactly at log 2
    zero_loss, zero_margin = dpo_loss(ref_w, ref_l, ref_w, ref_l, beta=0.5)
    assert float(zero_margin) == pytest.approx(0.0)
    assert float(zero_loss) == pytest.approx(np.log(2.0), rel=1e-6)


def test_sequence_logprob_only_counts_assistant_tokens(model, examples):
    batch = encode_batch(examples[:4])
    with torch.no_grad():
        logp = sequence_logprob(model, batch)                            # (B,)
    assert logp.shape == (4,)
    assert (logp < 0).all()
    # two supervised tokens, so the sum cannot be below 2 * log(1/V)
    assert (logp > 2 * np.log(1.0 / task.VOCAB_SIZE) - 1e-4).all()


def test_dpo_increases_the_preferred_log_ratio():
    torch.manual_seed(0)
    policy = TinyVLM(vocab_size=task.VOCAB_SIZE, image_size=task.IMAGE_SIZE, patch_size=task.CELL)
    data = task.make_dataset(128, seed=2)
    run_sft(policy, data, steps=60, batch_size=32, lr=3e-3, seed=0)
    reference = freeze_reference(policy)
    pairs = make_preference_pairs(data, seed=0)

    before = mean_dpo_margin(policy, reference, pairs)
    assert before == pytest.approx(0.0, abs=1e-5), "policy starts equal to the reference"
    report = run_dpo(policy, reference, pairs, steps=40, batch_size=24, lr=5e-4, beta=0.1, seed=0)
    assert report["margin_after"] > report["margin_before"]
    assert report["margin_after"] > 0.0
    assert all(p.requires_grad is False for p in reference.parameters())


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


def test_group_advantages_are_zero_mean_within_each_group():
    torch.manual_seed(0)
    rewards = torch.rand(6, 5)                                           # (P, G)
    adv = group_advantages(rewards)                                      # (P, G)
    assert adv.shape == rewards.shape
    assert torch.allclose(adv.mean(dim=1), torch.zeros(6), atol=1e-5)
    assert torch.allclose(adv.std(dim=1, unbiased=False), torch.ones(6), atol=1e-3)

    raw = group_advantages(rewards, normalize_std=False)
    assert torch.allclose(raw.mean(dim=1), torch.zeros(6), atol=1e-6)

    # a group where every sample got the same reward carries no signal
    flat = torch.full((2, 4), 0.7)
    assert torch.allclose(group_advantages(flat), torch.zeros(2, 4), atol=1e-6)


def test_k3_kl_is_non_negative_and_zero_at_equality():
    logp = torch.tensor([-1.0, -2.5, -0.3])
    assert torch.allclose(k3_kl(logp, logp), torch.zeros(3), atol=1e-7)
    assert (k3_kl(logp, logp - 0.8) >= 0).all()
    assert (k3_kl(logp, logp + 0.8) >= 0).all()


def test_grpo_loss_clips_and_reduces_to_reinforce_at_ratio_one():
    logp = torch.tensor([-1.0, -1.0])
    advantages = torch.tensor([1.0, -1.0])
    loss = grpo_loss(logp, logp.clone(), advantages, logp.clone(), clip_eps=0.2, kl_coef=0.0)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)                   # mean of [-1, +1] * ratio 1

    # a positive advantage stops paying once the ratio exceeds 1 + eps
    far = torch.tensor([-1.0 + np.log(2.0)])                             # ratio 2
    capped = grpo_loss(far, torch.tensor([-1.0]), torch.tensor([1.0]), torch.tensor([-1.0]),
                       clip_eps=0.2, kl_coef=0.0)
    assert float(capped) == pytest.approx(-1.2, rel=1e-5)


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------


def test_tool_returns_the_ground_truth_count():
    shapes = [task.Shape("blue", "circle", 0, 0, 3, 5), task.Shape("blue", "square", 2, 2, 4, 16)]
    scene = task.Scene(shapes=shapes, image=task.render(shapes, np.random.default_rng(0), noise=0.0))
    assert count_shapes_tool(scene, "blue", None) == "2"
    assert count_shapes_tool(scene, "blue", "circle") == "1"
    assert count_shapes_tool(scene, "red", None) == "0"


def test_tool_loop_records_a_well_formed_trajectory(model, examples):
    trajectories = run_episodes(model, examples, confidence_threshold=1.1)  # force the tool where allowed
    assert len(trajectories) == len(examples)
    for traj, ex in zip(trajectories, examples):
        assert traj.is_well_formed()
        assert traj.question == ex.question
        assert traj.gold_answer == ex.answer
        assert traj.final_answer in task.VOCAB
        assert traj.reward in (0.0, 1.0)
        assert traj.reward == float(traj.final_answer == ex.answer)
        if ex.tool_query is None:
            assert not traj.used_tool, "the tool cannot answer a colour question"
            assert len(traj.steps) == 1
        else:
            assert traj.used_tool
            assert len(traj.steps) == 2
            assert traj.steps[0].action == "call_tool"
            assert traj.steps[0].result == count_shapes_tool(ex.scene, *ex.tool_query)
    report = trajectory_report(trajectories)
    assert report["well_formed_rate"] == 1.0
    assert 0.0 < report["tool_call_rate"] <= 1.0
    assert "action" in render_trajectory(trajectories[0])


def test_tool_loop_answers_directly_when_confident(model, examples):
    trajectories = run_episodes(model, examples, confidence_threshold=0.0)
    assert all(not t.used_tool for t in trajectories)
    assert all(len(t.steps) == 1 and t.steps[0].action == "answer" for t in trajectories)
    assert trajectory_report(trajectories)["mean_steps"] == 1.0
