import numpy as np
import torch

torch.set_num_threads(1)
from torch import nn

from mlbook.reliability import uncertainty as unc


def test_heteroscedastic_gaussian_nll_matches_torch_reference():
    mu, y = torch.randn(10), torch.randn(10)
    log_var = torch.randn(10)
    ours = unc.heteroscedastic_gaussian_nll(mu, log_var, y)
    ref = nn.functional.gaussian_nll_loss(mu, y, log_var.exp(), full=False, eps=0.0)
    assert torch.isclose(ours, ref, atol=1e-6)


def test_heteroscedastic_nll_gradient_wrt_log_var():
    mu, y = torch.zeros(1), torch.tensor([2.0])
    lv = torch.zeros(1, requires_grad=True)
    unc.heteroscedastic_gaussian_nll(mu, lv, y).backward()
    # d/dlogvar [0.5 logvar + r^2/(2 e^logvar)] = 0.5 - r^2/(2 e^logvar) = 0.5 - 2 = -1.5
    assert torch.isclose(lv.grad, torch.tensor([-1.5]))


def test_entropy_predictive_expected_and_mutual_information():
    P = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # M=2, N=1, K=2 : members disagree confidently
    assert np.isclose(unc.predictive_entropy(P)[0], np.log(2), atol=1e-6)
    assert np.isclose(unc.expected_entropy(P)[0], 0.0, atol=1e-6)
    assert np.isclose(unc.mutual_information(P)[0], np.log(2), atol=1e-6)
    P2 = np.array([[[0.5, 0.5]], [[0.5, 0.5]]])  # agree but uncertain: aleatoric only
    assert np.isclose(unc.mutual_information(P2)[0], 0.0, atol=1e-6)
    assert np.isclose(unc.entropy(np.array([0.5, 0.5])), np.log(2), atol=1e-6)


def test_mc_dropout_predict_is_stochastic_and_normalised():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 3))
    x = torch.randn(5, 4)
    P = unc.mc_dropout_predict(model, x, n_samples=8)
    assert P.shape == (8, 5, 3) and np.allclose(P.sum(-1), 1.0)
    assert P.std(axis=0).max() > 0  # dropout was active
    assert not model.training and not any(m.training for m in model.modules())


def test_ensemble_predict_shape():
    torch.manual_seed(0)
    models = [nn.Linear(4, 3) for _ in range(3)]
    P = unc.ensemble_predict(models, torch.randn(6, 4))
    assert P.shape == (3, 6, 3) and np.allclose(P.sum(-1), 1.0)
