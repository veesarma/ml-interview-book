"""PPO for a language model: shaped rewards, GAE, the clipped surrogate, value loss, KL controller,
and an end-to-end run on the toy LM."""
import copy

import torch

from mlbook.posttrain.chat_template import ToyTokenizer, prompt_ids
from mlbook.posttrain.ppo_lm import (
    AdaptiveKLController,
    TinyCritic,
    gae,
    last_response_index,
    masked_entropy,
    ppo_clip_loss,
    shaped_rewards,
    train_ppo,
    value_loss,
)
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig
from mlbook.posttrain.verifiers import target_token_reward


def test_last_response_index_finds_final_response_token():
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])
    assert last_response_index(mask).tolist() == [2, 2, 3]


def test_shaped_rewards_places_score_and_kl():
    scores = torch.tensor([2.0, -1.0])
    logp_pi = torch.tensor([[-1.0, -0.5, -2.0], [-1.0, -1.0, -1.0]])
    logp_ref = torch.tensor([[-1.2, -0.5, -1.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    beta = 0.1
    R = shaped_rewards(scores, logp_pi, logp_ref, mask, beta)
    assert R.shape == (2, 3)
    # row 0: KL terms are -beta * (logp_pi - logp_ref), the score lands on the last response token (t=2)
    expected0 = torch.tensor([-0.1 * 0.2, 0.0, -0.1 * (-1.0) + 2.0])
    assert torch.allclose(R[0], expected0, atol=1e-6)
    # row 1: policy == reference so every KL term is zero; the score lands at t=1 (mask ends there)
    assert torch.allclose(R[1], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)
    # with beta = 0 only the sequence score survives
    R0 = shaped_rewards(scores, logp_pi, logp_ref, mask, beta=0.0)
    assert torch.allclose(R0, torch.tensor([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0]]), atol=1e-6)


def test_gae_matches_brute_force_and_mc_return():
    torch.manual_seed(0)
    B, L = 3, 6
    rewards = torch.randn(B, L)
    values = torch.randn(B, L)
    mask = torch.ones(B, L)
    gamma, lam = 1.0, 0.95
    adv, ret = gae(rewards, values, mask, gamma, lam)
    # brute force: A_t = sum_l (gamma*lam)^l delta_{t+l}, with V_{L} = 0
    next_values = torch.cat([values[:, 1:], torch.zeros(B, 1)], dim=1)
    delta = rewards + gamma * next_values - values
    brute = torch.zeros(B, L)
    for t in range(L):
        for l in range(L - t):
            brute[:, t] += (gamma * lam) ** l * delta[:, t + l]
    assert torch.allclose(adv, brute, atol=1e-5)
    assert torch.allclose(ret, adv + values, atol=1e-6)
    # at lambda = gamma = 1 the returns are the Monte-Carlo reward-to-go
    adv1, ret1 = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    mc = rewards.flip(1).cumsum(1).flip(1)
    assert torch.allclose(ret1, mc, atol=1e-5)


def test_gae_masks_padding_after_the_response():
    rewards = torch.tensor([[1.0, 1.0, 5.0]])
    values = torch.zeros(1, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # position 2 is padding
    adv, _ = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    # the padded position contributes nothing and does not leak into t = 1
    assert adv[0, 2] == 0.0
    assert torch.allclose(adv[0, :2], torch.tensor([2.0, 1.0]), atol=1e-6)


def test_ppo_clip_loss_gradient_and_clipping():
    adv = torch.tensor([[1.0, -1.0]])
    mask = torch.ones(1, 2)
    logp_old = torch.tensor([[-1.0, -1.0]])
    # at ratio == 1 the loss is -mean(A) and the gradient is the vanilla policy gradient
    logp_new = logp_old.clone().requires_grad_(True)
    loss, clip_frac = ppo_clip_loss(logp_new, logp_old, adv, mask, clip_eps=0.2)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)  # (-1 + 1) / 2
    assert clip_frac == 0.0
    loss.backward()
    assert torch.allclose(logp_new.grad, -adv / 2, atol=1e-6)

    # A > 0 and ratio beyond 1 + eps: the term is flat, so no gradient flows to that token
    logp_hi = torch.tensor([[-1.0 + 0.5, -1.0]], requires_grad=True)  # ratio = e^0.5 = 1.65 > 1.2
    loss_hi, clip_hi = ppo_clip_loss(logp_hi, logp_old, adv, mask, clip_eps=0.2)
    loss_hi.backward()
    assert logp_hi.grad[0, 0] == 0.0
    assert clip_hi == 0.5  # one of the two tokens is clipped

    # A < 0 and ratio below 1 - eps: also flat
    logp_lo = torch.tensor([[-1.0, -1.0 - 0.5]], requires_grad=True)
    loss_lo, _ = ppo_clip_loss(logp_lo, logp_old, adv, mask, clip_eps=0.2)
    loss_lo.backward()
    assert logp_lo.grad[0, 1] == 0.0

    # moving the *wrong* way is never clipped: A > 0 with a shrinking ratio still has gradient
    logp_wrong = torch.tensor([[-1.0 - 0.5, -1.0]], requires_grad=True)
    loss_w, _ = ppo_clip_loss(logp_wrong, logp_old, adv, mask, clip_eps=0.2)
    loss_w.backward()
    assert logp_wrong.grad[0, 0] < 0


def test_ppo_clip_loss_is_a_lower_bound_on_the_unclipped_surrogate():
    torch.manual_seed(0)
    logp_old = torch.randn(4, 5)
    logp_new = logp_old + 0.4 * torch.randn(4, 5)
    adv = torch.randn(4, 5)
    mask = torch.ones(4, 5)
    clipped, _ = ppo_clip_loss(logp_new, logp_old, adv, mask, clip_eps=0.2)
    unclipped = -((torch.exp(logp_new - logp_old) * adv) * mask).sum() / mask.sum()
    assert clipped >= unclipped - 1e-6  # loss is negated, so the clipped objective is the lower bound


def test_value_loss_clipping():
    old_values = torch.zeros(1, 2)
    returns = torch.tensor([[1.0, 1.0]])
    mask = torch.ones(1, 2)
    # inside the trust region the loss is the plain squared error
    v_in = torch.tensor([[0.1, 0.1]])
    assert torch.allclose(value_loss(v_in, old_values, returns, mask, clip_eps=0.2),
                          0.5 * (v_in - returns).pow(2).mean(), atol=1e-6)
    # far outside, the max picks the clipped branch, which is the larger error here
    v_out = torch.tensor([[5.0, 5.0]])
    clipped_pred = old_values + torch.clamp(v_out - old_values, -0.2, 0.2)
    expected = 0.5 * torch.max((v_out - returns) ** 2, (clipped_pred - returns) ** 2).mean()
    assert torch.allclose(value_loss(v_out, old_values, returns, mask, clip_eps=0.2), expected, atol=1e-6)


def test_masked_entropy_matches_uniform_and_respects_mask():
    V = 8
    logits = torch.zeros(1, 2, V)  # uniform over V
    mask = torch.tensor([[1.0, 0.0]])
    ent = masked_entropy(logits, mask)
    assert torch.allclose(ent, torch.log(torch.tensor(float(V))), atol=1e-6)
    # a peaked distribution at the masked position must not change the answer
    logits2 = logits.clone()
    logits2[0, 1, 0] = 50.0
    assert torch.allclose(masked_entropy(logits2, mask), ent, atol=1e-6)


def test_kl_controller_moves_beta_toward_target():
    ctl = AdaptiveKLController(beta=0.1, target=1.0, horizon=10)
    # KL above target raises beta
    assert ctl.update(2.0) > 0.1
    ctl2 = AdaptiveKLController(beta=0.1, target=1.0, horizon=10)
    assert ctl2.update(0.1) < 0.1  # KL below target lowers beta
    # the proportional error is clipped to +/- 0.2 so one step can move beta by at most 2 %
    ctl3 = AdaptiveKLController(beta=1.0, target=1.0, horizon=10)
    assert abs(ctl3.update(1000.0) - 1.02) < 1e-9
    ctl4 = AdaptiveKLController(beta=1.0, target=1.0, horizon=10)
    assert abs(ctl4.update(0.0) - 0.98) < 1e-9


def _ppo_setup(seed=0, n_prompts=32):
    torch.manual_seed(seed)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=2, max_len=32)
    policy = TinyCausalLM(cfg)
    ref = copy.deepcopy(policy)
    critic = TinyCritic(cfg)
    prompt = torch.tensor([prompt_ids([{"role": "user", "content": "hello"}], tok)]).repeat(n_prompts, 1)
    return tok, policy, ref, critic, prompt


def test_train_ppo_increases_reward_with_bounded_kl():
    """Programmatic reward: emit the token `good` as often as possible in 4 response tokens."""
    tok, policy, ref, critic, prompts = _ppo_setup(seed=0)
    target = tok.id_of["good"]
    reward_fn = lambda t, m: target_token_reward(t, m, target)
    history = train_ppo(policy, ref, critic, prompts, reward_fn, tok.eos_id,
                        steps=20, max_new_tokens=4, beta=0.02, lr=3e-3)
    assert len(history) == 20
    first = sum(h.reward for h in history[:5]) / 5
    last = sum(h.reward for h in history[-5:]) / 5
    assert first < 0.6            # a random policy hits the target token rarely (1 in 54 per token)
    assert last > 5 * first       # reward rises by a large factor
    assert last > 1.5             # more than one target token per response on average
    # the KL to the reference grows but stays finite, and the reference is untouched
    assert all(h.kl == h.kl for h in history)  # no NaN
    assert max(h.kl for h in history) < 40.0
    assert all(not p.requires_grad for p in ref.parameters())


def test_ppo_kl_penalty_constrains_drift():
    """A large beta buys less reward and keeps the policy nearer the reference than a small beta."""
    runs = {}
    for beta in (0.02, 2.0):
        tok, policy, ref, critic, prompts = _ppo_setup(seed=0)
        target = tok.id_of["good"]
        reward_fn = lambda t, m: target_token_reward(t, m, target)
        history = train_ppo(policy, ref, critic, prompts, reward_fn, tok.eos_id,
                            steps=20, max_new_tokens=4, beta=beta, lr=3e-3)
        runs[beta] = history
    kl_small = max(h.kl for h in runs[0.02])
    kl_large = max(h.kl for h in runs[2.0])
    reward_small = sum(h.reward for h in runs[0.02][-5:]) / 5
    reward_large = sum(h.reward for h in runs[2.0][-5:]) / 5
    assert kl_large < kl_small          # the penalty does hold the policy back
    assert reward_large < reward_small  # and that costs proxy reward
