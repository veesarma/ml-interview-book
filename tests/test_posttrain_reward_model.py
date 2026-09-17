"""Reward model: Bradley-Terry loss vs. logistic regression, margin, and learning a synthetic preference."""
import torch
import torch.nn.functional as F

from mlbook.posttrain.chat_template import ToyTokenizer, tokenize_chat
from mlbook.posttrain.reward_model import (
    TinyRewardModel,
    bradley_terry_loss,
    bradley_terry_prob,
    pairwise_accuracy,
    train_reward_model,
)
from mlbook.posttrain.toy_lm import ToyLMConfig


def test_bradley_terry_loss_equals_logistic_regression_on_gap():
    torch.manual_seed(0)
    r_w, r_l = torch.randn(16), torch.randn(16)
    loss = bradley_terry_loss(r_w, r_l)
    # logistic regression with feature = gap, weight 1, label 1
    manual = F.binary_cross_entropy_with_logits(r_w - r_l, torch.ones(16))
    assert torch.allclose(loss, manual)
    assert torch.allclose(bradley_terry_prob(r_w, r_l), torch.sigmoid(r_w - r_l))
    # symmetric: swapping the pair gives the complementary probability
    assert torch.allclose(bradley_terry_prob(r_w, r_l) + bradley_terry_prob(r_l, r_w), torch.ones(16))


def test_margin_increases_loss_until_gap_exceeds_it():
    r_w, r_l = torch.tensor([2.0]), torch.tensor([0.0])
    assert bradley_terry_loss(r_w, r_l, margin=1.0) > bradley_terry_loss(r_w, r_l, margin=0.0)
    # gradient of the plain loss w.r.t. r_w is -(1 - sigma(gap))
    r_w.requires_grad_(True)
    bradley_terry_loss(r_w, r_l).backward()
    assert torch.allclose(r_w.grad, -(1 - torch.sigmoid(r_w.detach() - r_l)))


def _pair_tensors(tok, prompts, chosen, rejected, T=16):
    def rows(answers):
        ids_list = []
        for p, a in zip(prompts, answers):
            ids, _ = tokenize_chat([{"role": "user", "content": p}, {"role": "assistant", "content": a}], tok)
            ids_list.append(ids)
        lengths = torch.tensor([len(i) for i in ids_list])
        out = torch.full((len(ids_list), T), tok.pad_id)
        for k, i in enumerate(ids_list):
            out[k, : len(i)] = torch.tensor(i)
        return out, lengths
    c, cl = rows(chosen)
    r, rl = rows(rejected)
    return c, cl, r, rl


def test_reward_model_learns_synthetic_preference():
    """Preference rule: answers containing 'yes' beat answers containing 'no'."""
    torch.manual_seed(0)
    tok = ToyTokenizer()
    prompts = ["are you sure ?", "is the cat blue ?", "hello", "is 2 + 2 = 4 ?", "good dog ?", "red or blue ?"] * 4
    chosen = ["yes I think so", "yes", "yes hello", "yes sure", "yes great", "yes red"] * 4
    rejected = ["no", "no not sure", "no sorry", "no", "no bad", "no blue"] * 4
    c, cl, r, rl = _pair_tensors(tok, prompts, chosen, rejected)
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_len=32)
    rm = TinyRewardModel(cfg)
    with torch.no_grad():
        acc_before = pairwise_accuracy(rm(c, cl), rm(r, rl))
    losses = train_reward_model(rm, c, cl, r, rl, steps=60, lr=5e-3, batch_size=24)
    with torch.no_grad():
        r_w, r_l = rm(c, cl), rm(r, rl)
    assert r_w.shape == (24,)
    assert losses[-1] < losses[0]
    assert pairwise_accuracy(r_w, r_l) == 1.0
    assert pairwise_accuracy(r_w, r_l) >= acc_before
    # the scalar head reads the *last real* token: padding after `lengths` must not change the reward
    c2 = c.clone()
    c2[:, 12:] = tok.unk_id  # corrupt only padding positions (all chosen rows are shorter than 12 tokens)
    with torch.no_grad():
        assert torch.allclose(rm(c2, cl), r_w, atol=1e-5)
