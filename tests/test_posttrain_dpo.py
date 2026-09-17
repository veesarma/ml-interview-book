"""DPO and relatives: closed-form checks, gradient sign, and a training step on the toy LM."""
import copy

import torch
import torch.nn.functional as F

from mlbook.posttrain.chat_template import ToyTokenizer, tokenize_chat
from mlbook.posttrain.dpo import (
    dpo_loss,
    dpo_train_step,
    ipo_loss,
    kto_loss,
    kto_reference_point,
    orpo_loss,
    simpo_loss,
)
from mlbook.posttrain.toy_lm import TinyCausalLM, ToyLMConfig, sequence_log_prob


def test_dpo_loss_closed_form_and_implicit_rewards():
    pi_c, pi_r = torch.tensor([-2.0]), torch.tensor([-5.0])
    ref_c, ref_r = torch.tensor([-3.0]), torch.tensor([-3.0])
    beta = 0.5
    loss, r_c, r_r = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))  # 0.5 * (1 - (-2)) = 1.5
    assert torch.allclose(loss, -F.logsigmoid(margin))
    assert torch.allclose(r_c, torch.tensor([0.5])) and torch.allclose(r_r, torch.tensor([-1.0]))
    # at initialisation (pi == ref) the loss is log 2
    loss0, _, _ = dpo_loss(ref_c, ref_r, ref_c, ref_r, beta)
    assert torch.allclose(loss0, torch.log(torch.tensor(2.0)))


def test_dpo_gradient_sign_and_weighting():
    """d loss / d pi_c = -beta * sigma(-margin) < 0 (push chosen up); d/d pi_r = +beta * sigma(-margin)."""
    beta = 0.3
    pi_c = torch.tensor([-1.0, -4.0], requires_grad=True)
    pi_r = torch.tensor([-1.0, -1.0], requires_grad=True)
    ref_c, ref_r = torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, -1.0])
    loss, _, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta)
    loss.backward()
    assert (pi_c.grad < 0).all() and (pi_r.grad > 0).all()
    # second pair has the *wrong* ordering (margin < 0): its gradient weight sigma(-margin) is larger
    assert pi_c.grad[1].abs() > pi_c.grad[0].abs()
    margins = beta * ((pi_c.detach() - ref_c) - (pi_r.detach() - ref_r))
    expected = -beta * torch.sigmoid(-margins) / 2  # mean over B=2
    assert torch.allclose(pi_c.grad, expected)


def test_ipo_simpo_orpo_kto_basic_properties():
    pi_c, pi_r = torch.tensor([-2.0, -3.0]), torch.tensor([-4.0, -2.0])
    ref_c, ref_r = torch.tensor([-3.0, -3.0]), torch.tensor([-3.0, -3.0])
    lens_c, lens_r = torch.tensor([2.0, 3.0]), torch.tensor([4.0, 2.0])
    # IPO is minimised when the gap equals 1/(2 tau)
    tau = 0.25
    gap = torch.tensor([1.0 / (2 * tau), 1.0 / (2 * tau)])
    assert torch.allclose(ipo_loss(ref_c + gap, ref_r, ref_c, ref_r, tau), torch.tensor(0.0))
    assert ipo_loss(pi_c, pi_r, ref_c, ref_r, tau) > 0
    # SimPO needs no reference and increases with the target margin gamma
    s0 = simpo_loss(pi_c, pi_r, lens_c, lens_r, beta=2.0, gamma=0.0)
    s1 = simpo_loss(pi_c, pi_r, lens_c, lens_r, beta=2.0, gamma=1.0)
    assert s1 > s0
    # ORPO: pure NLL term when lam = 0
    assert torch.allclose(orpo_loss(pi_c, pi_r, lens_c, lens_r, lam=0.0), -(pi_c / lens_c).mean())
    assert orpo_loss(pi_c, pi_r, lens_c, lens_r, lam=1.0) != orpo_loss(pi_c, pi_r, lens_c, lens_r, lam=0.0)
    # KTO: at initialisation (pi == ref) the reward and z0 are both 0, every value is sigma(0) = 0.5,
    # so the loss is mean(lambda - 0.5) = 0.5 whatever the labels are.
    pi = torch.tensor([-1.0, -2.0, -3.0])
    assert torch.allclose(kto_loss(pi, pi, torch.tensor([1.0, 0.0, 1.0]), beta=0.1), torch.tensor(0.5))
    assert torch.allclose(kto_reference_point(pi, pi, beta=0.1), torch.tensor(0.0))
    # z0 is the batch-mean implicit reward clamped at 0, and is detached (no gradient path through it).
    pi_grad = torch.tensor([-1.0, -1.0], requires_grad=True)
    assert kto_reference_point(pi_grad, torch.tensor([-2.0, -2.0]), beta=0.5).requires_grad is False
    assert torch.allclose(kto_reference_point(pi_grad, torch.tensor([-2.0, -2.0]), beta=0.5), torch.tensor(0.5))
    # A response *above* the reference point is what a desirable label wants and an undesirable label
    # does not: at the same reward the two labels give complementary values. Pass z0 explicitly,
    # because the in-batch default would equal the reward itself when every row is identical.
    ref = torch.tensor([-2.0, -2.0])
    pi_hi = torch.tensor([-1.0, -1.0])  # implicit reward = beta * 1.0 = 1.0
    ld = kto_loss(pi_hi, ref, torch.tensor([1.0, 1.0]), beta=1.0, z0=0.0)
    lu = kto_loss(pi_hi, ref, torch.tensor([0.0, 0.0]), beta=1.0, z0=0.0)
    assert ld < lu
    assert torch.allclose(ld, 1.0 - torch.sigmoid(torch.tensor(1.0)))
    assert torch.allclose(lu, 1.0 - torch.sigmoid(torch.tensor(-1.0)))
    # Loss aversion: weighting undesirable examples more (lambda_u > lambda_d) raises their cost.
    assert kto_loss(pi_hi, ref, torch.tensor([0.0, 0.0]), beta=1.0, z0=0.0, lambda_u=2.0) > lu


def _pairs(tok):
    prompts = ["2 + 2 =", "1 + 1 =", "3 + 4 =", "0 + 5 ="]
    good = ["4", "2", "7", "5"]
    bad = ["5", "3", "8", "6"]
    rows = lambda answers: [tokenize_chat([{"role": "user", "content": p}, {"role": "assistant", "content": a}], tok) for p, a in zip(prompts, answers)]
    c, r = rows(good), rows(bad)
    to_t = lambda ex: (torch.tensor([e[0] for e in ex]), torch.tensor([e[1] for e in ex], dtype=torch.float))
    return to_t(c), to_t(r)


def test_dpo_train_step_raises_preferred_log_ratio():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    cfg = ToyLMConfig(vocab_size=tok.vocab_size, d_model=32, n_heads=2, n_layers=1, max_len=32)
    policy = TinyCausalLM(cfg)
    ref = copy.deepcopy(policy)
    (chosen, m_c), (rejected, m_r) = _pairs(tok)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    beta = 0.1
    with torch.no_grad():
        ref_c, ref_r = sequence_log_prob(ref, chosen, m_c), sequence_log_prob(ref, rejected, m_r)
    first_loss, first_margin = dpo_train_step(policy, ref, opt, chosen, m_c, rejected, m_r, beta)
    assert abs(first_loss - float(torch.log(torch.tensor(2.0)))) < 1e-5  # policy == ref at step 0
    assert abs(first_margin) < 1e-5
    for _ in range(30):
        loss, margin = dpo_train_step(policy, ref, opt, chosen, m_c, rejected, m_r, beta)
    with torch.no_grad():
        pi_c, pi_r = sequence_log_prob(policy, chosen, m_c), sequence_log_prob(policy, rejected, m_r)
    assert loss < first_loss and margin > 0.5
    assert ((pi_c - ref_c) > 0).all()  # chosen log-ratio rose on every pair
    assert ((pi_r - ref_r) < 0).all()  # rejected log-ratio fell on every pair
