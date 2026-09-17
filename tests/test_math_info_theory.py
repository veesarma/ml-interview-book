import numpy as np
import torch
from scipy import stats as sps
from scipy.special import rel_entr

from mlbook.math import info_theory as it


def test_entropy_known_values():
    assert np.isclose(it.entropy(np.array([0.5, 0.5]), base=2), 1.0)
    assert np.isclose(it.entropy(np.array([1.0, 0.0])), 0.0)
    p = np.random.dirichlet(np.ones(6))
    assert np.isclose(it.entropy(p), sps.entropy(p))


def test_kl_nonneg_asymmetric_and_matches_scipy():
    p = np.random.dirichlet(np.ones(5))
    q = np.random.dirichlet(np.ones(5))
    kl_pq = it.kl_divergence(p, q)
    assert kl_pq >= 0
    assert np.isclose(kl_pq, np.sum(rel_entr(p, q)))
    assert not np.isclose(kl_pq, it.kl_divergence(q, p))
    assert np.isclose(it.cross_entropy(p, q) - it.entropy(p), kl_pq)
    assert np.isclose(it.kl_divergence(p, p), 0.0)


def test_cross_entropy_matches_torch_nll():
    logits = np.random.randn(4, 7)
    y = np.array([0, 3, 6, 2])
    q = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # (4, 7)
    onehot = np.eye(7)[y]  # (4, 7)
    ce = it.cross_entropy(onehot, q)  # (4,)
    ref = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(y), reduction="none").numpy()
    assert np.allclose(ce, ref)


def test_js_bounded_symmetric():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.isclose(it.js_divergence(p, q), np.log(2))
    a, b = np.random.dirichlet(np.ones(4)), np.random.dirichlet(np.ones(4))
    assert np.isclose(it.js_divergence(a, b), it.js_divergence(b, a))


def test_mutual_information_and_conditional_entropy():
    joint = np.array([[0.25, 0.25], [0.25, 0.25]])  # independent
    assert np.isclose(it.mutual_information(joint), 0.0)
    joint = np.array([[0.5, 0.0], [0.0, 0.5]])  # Y = X
    assert np.isclose(it.mutual_information(joint), np.log(2))
    assert np.isclose(it.conditional_entropy(joint), 0.0)
    joint = np.random.dirichlet(np.ones(12)).reshape(3, 4)
    H_y = it.entropy(joint.sum(axis=0))
    assert np.isclose(it.mutual_information(joint), H_y - it.conditional_entropy(joint))


def test_gaussian_kl_matches_torch():
    mu1, mu2 = np.random.randn(3), np.random.randn(3)
    var1, var2 = np.exp(np.random.randn(3)), np.exp(np.random.randn(3))
    ours = it.gaussian_kl(mu1, var1, mu2, var2)
    p = torch.distributions.Normal(torch.tensor(mu1), torch.tensor(var1).sqrt())
    q = torch.distributions.Normal(torch.tensor(mu2), torch.tensor(var2).sqrt())
    assert np.isclose(ours, torch.distributions.kl_divergence(p, q).sum().item())


def test_perplexity_and_bpb():
    nll = np.log(np.array([4.0, 4.0, 4.0]))  # each token had prob 1/4
    assert np.isclose(it.perplexity(nll), 4.0)
    # 3 tokens of 2 bits each over 3 bytes -> 2 bits per byte
    assert np.isclose(it.bits_per_byte(nll, n_bytes=3), 2.0)


def test_infonce_matches_torch_cross_entropy():
    za, zb = np.random.randn(8, 16), np.random.randn(8, 16)
    ours = it.infonce_loss(za, zb, temperature=0.1)
    a = torch.nn.functional.normalize(torch.tensor(za), dim=1)
    b = torch.nn.functional.normalize(torch.tensor(zb), dim=1)
    logits = a @ b.T / 0.1
    labels = torch.arange(8)
    ref = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels))
    assert np.isclose(ours, ref.item())
    # lower bound on MI: I >= log N - L must hold with a finite value
    assert np.log(8) - ours < np.log(8)


def test_entropy_bonus_uniform_is_log_k():
    logits = np.zeros((5, 10))
    assert np.isclose(it.entropy_bonus(logits), np.log(10))


def test_forward_vs_reverse_kl_fit():
    xs = np.linspace(-8, 8, 401)
    p = 0.5 * np.exp(-0.5 * ((xs + 3) / 0.7) ** 2) + 0.5 * np.exp(-0.5 * ((xs - 3) / 0.7) ** 2)
    p /= p.sum()
    mu_f, sig_f = it.fit_gaussian_to_mixture_kl(xs, p, "forward")
    mu_r, sig_r = it.fit_gaussian_to_mixture_kl(xs, p, "reverse")
    assert abs(mu_f) < 0.5 and sig_f > 2.0  # mode-covering: wide, centred
    assert abs(abs(mu_r) - 3.0) < 0.5 and sig_r < 1.2  # mode-seeking: one mode
