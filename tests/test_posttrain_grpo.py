"""GRPO and RLVR: group-relative advantages, the k3 KL estimator, the clipped group surrogate,
the programmatic verifiers, and an end-to-end run where verified accuracy improves."""
import copy

import torch

from mlbook.posttrain.chat_template import ToyTokenizer
from mlbook.posttrain.grpo import group_relative_advantages, grpo_loss, k3_kl, train_grpo
from mlbook.posttrain.sft import pack_examples, train_sft
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig, sample
from mlbook.posttrain.verifiers import (
    all_arithmetic_pairs,
    arithmetic_answer,
    arithmetic_prompt_batch,
    arithmetic_reward,
    make_arithmetic_sft_examples,
    response_text,
    target_token_reward,
)


def test_group_relative_advantages_standardises_each_group():
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
    adv = group_relative_advantages(rewards)
    assert adv.shape == (2, 4)
    # each group has mean 0 and (population) std 1
    assert torch.allclose(adv.mean(dim=1), torch.zeros(2), atol=1e-4)
    assert torch.allclose(adv.std(dim=1, unbiased=False), torch.ones(2), atol=1e-3)
    # correct responses get positive advantage, incorrect negative
    assert (adv[0, [0, 3]] > 0).all() and (adv[0, [1, 2]] < 0).all()
    # the rarer the success, the larger its advantage
    assert adv[1, 3] > adv[0, 0]


def test_group_with_identical_rewards_has_zero_advantage():
    """All-correct or all-wrong groups produce no gradient signal: the DAPO 'zero advantage' problem."""
    for r in (torch.ones(1, 6), torch.zeros(1, 6)):
        adv = group_relative_advantages(r)
        assert torch.allclose(adv, torch.zeros(1, 6), atol=1e-6)


def test_advantages_without_std_normalisation_are_centred_only():
    """Dr. GRPO drops the std divisor, which otherwise up-weights low-variance groups."""
    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    centred = group_relative_advantages(rewards, normalize_std=False)
    assert torch.allclose(centred, rewards - rewards.mean(), atol=1e-6)
    scaled = group_relative_advantages(rewards, normalize_std=True)
    # the std of this group is 0.433, so dividing by it magnifies the advantages
    assert scaled.abs().max() > centred.abs().max()


def test_k3_kl_is_non_negative_and_unbiased():
    torch.manual_seed(0)
    V = 12
    logits_p = torch.randn(4000, V)
    logits_q = logits_p + 0.3 * torch.randn(1, V)  # a nearby distribution
    logp_p = torch.log_softmax(logits_p, dim=-1)
    logp_q = torch.log_softmax(logits_q, dim=-1)
    # sample actions from p and evaluate the estimator on them
    actions = torch.multinomial(logp_p.exp(), num_samples=1).squeeze(-1)
    lp = logp_p.gather(1, actions.unsqueeze(1)).squeeze(1)
    lq = logp_q.gather(1, actions.unsqueeze(1)).squeeze(1)
    est = k3_kl(lp, lq)
    assert (est >= 0).all()                       # k3 is non-negative by construction
    true_kl = (logp_p.exp() * (logp_p - logp_q)).sum(dim=-1).mean()
    assert abs(float(est.mean()) - float(true_kl)) < 0.02   # unbiased under samples from p
    # identical distributions give exactly zero
    assert torch.allclose(k3_kl(lp, lp), torch.zeros_like(lp), atol=1e-6)


def test_grpo_loss_shapes_signs_and_clipping():
    logp_old = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    logp_ref = logp_old.clone()
    mask = torch.ones(2, 2)
    adv = torch.tensor([1.0, -1.0])  # one good response, one bad
    logp_new = logp_old.clone().requires_grad_(True)
    loss = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, clip_eps=0.2, beta=0.0)
    assert loss.dim() == 0
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)  # advantages cancel at ratio 1
    loss.backward()
    # the positively-advantaged response is pushed up, the negative one down
    assert (logp_new.grad[0] < 0).all() and (logp_new.grad[1] > 0).all()

    # the KL term is non-negative and adds to the loss when the policy leaves the reference
    drifted = logp_old - 0.5
    with_kl = grpo_loss(drifted, logp_old, logp_ref, torch.zeros(2), mask, beta=1.0)
    assert with_kl > 0

    # clip-higher (DAPO): a larger upper bound lets positively-advantaged tokens keep their gradient
    logp_up = (logp_old + 0.3).requires_grad_(True)  # ratio = 1.35, beyond 1.2 but inside 1.5
    tight = grpo_loss(logp_up, logp_old, logp_ref, torch.tensor([1.0, 0.0]), mask, clip_eps=0.2, beta=0.0)
    loose = grpo_loss(logp_up, logp_old, logp_ref, torch.tensor([1.0, 0.0]), mask,
                      clip_eps=0.2, clip_eps_high=0.5, beta=0.0)
    assert loose < tight  # the unclipped surrogate is larger, so the (negated) loss is smaller


def test_grpo_token_level_vs_sequence_level_aggregation():
    """Token-level aggregation weights a long response more than a short one; sequence-level does not."""
    logp_old = torch.zeros(2, 4)
    logp_ref = torch.zeros(2, 4)
    logp_new = torch.full((2, 4), 0.1)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])  # 4 tokens vs 1 token
    seq = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, beta=0.0, token_level=False)
    tok = grpo_loss(logp_new, logp_old, logp_ref, adv, mask, beta=0.0, token_level=True)
    # sequence-level averages per response first, so the +1 and -1 responses cancel exactly
    assert torch.allclose(seq, torch.tensor(0.0), atol=1e-6)
    # token-level lets the 4-token positive response dominate the 1-token negative one
    assert tok < -1e-3


def test_arithmetic_verifier_scores_exact_answers_only():
    tok = ToyTokenizer()
    assert arithmetic_answer(3, 4) == "7"
    assert all(a + b <= 9 for a, b in all_arithmetic_pairs())
    prompts, answers = arithmetic_prompt_batch([(1, 2), (3, 4)], tok)
    assert prompts.shape[0] == 2 and answers == ["3", "7"]
    # build responses by hand: one right, one wrong
    right = torch.cat([prompts[:1], torch.tensor([[tok.id_of["3"], tok.eos_id]])], dim=1)
    wrong = torch.cat([prompts[1:], torch.tensor([[tok.id_of["9"], tok.eos_id]])], dim=1)
    tokens = torch.cat([right, wrong], dim=0)
    mask = torch.zeros_like(tokens, dtype=torch.float)
    mask[:, prompts.shape[1]:] = 1.0
    r = arithmetic_reward(tokens, mask, answers, tok)
    assert r.tolist() == [1.0, 0.0]
    assert response_text(tokens[0], mask[0], tok) == "3"


def test_target_token_reward_counts_only_response_tokens():
    tokens = torch.tensor([[5, 7, 7, 7]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])  # the first and last 7 are outside the response
    assert target_token_reward(tokens, mask, target_id=7).tolist() == [2.0]


def _sft_policy_on_arithmetic(seed=0):
    torch.manual_seed(seed)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=2, max_len=32)
    model = TinyCausalLM(cfg)
    examples = make_arithmetic_sft_examples(tok, noise_rate=0.0, repeats=1)
    train_sft(model, [pack_examples(examples, max_len=20, pad_id=tok.pad_id)], epochs=80, lr=1e-2)
    return tok, model


def _sampled_accuracy(model, prompts, answers, tok, repeats=8):
    prompts_rep = prompts.repeat(repeats, 1)
    tokens, mask = sample(model, prompts_rep, 2, tok.eos_id)
    return float(arithmetic_reward(tokens, mask, answers * repeats, tok).mean())


def test_train_grpo_improves_verified_accuracy():
    """RLVR on single-digit addition: GRPO sharpens a partially-correct SFT policy, KL stays bounded."""
    tok, model = _sft_policy_on_arithmetic(seed=0)
    prompts, answers = arithmetic_prompt_batch(all_arithmetic_pairs(), tok)
    before = _sampled_accuracy(model, prompts, answers, tok)
    assert 0.2 < before < 0.75  # the SFT policy is right some of the time, with room to improve

    ref = copy.deepcopy(model)

    def reward_fn(tokens, mask, prompt_index):
        return arithmetic_reward(tokens, mask, [answers[i] for i in prompt_index.tolist()], tok)

    history = train_grpo(model, ref, prompts, reward_fn, tok.eos_id,
                         group_size=8, steps=25, max_new_tokens=2, lr=3e-3, beta=0.04)
    assert len(history) == 25
    after = _sampled_accuracy(model, prompts, answers, tok)
    first = sum(h.mean_reward for h in history[:3]) / 3
    last = sum(h.mean_reward for h in history[-3:]) / 3
    assert last > first + 0.05          # the group reward improves during training
    assert after > before + 0.05        # and so does freshly sampled accuracy
    assert max(h.kl for h in history) < 5.0   # the k3 KL to the reference stays small
    assert all(not p.requires_grad for p in ref.parameters())
    # some groups are all-correct or all-wrong and contribute no gradient
    assert 0.0 <= history[-1].frac_all_same <= 1.0
