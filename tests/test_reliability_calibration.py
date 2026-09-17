import numpy as np
import torch

torch.set_num_threads(1)

from mlbook.reliability import calibration as cal


def test_expected_calibration_error_hand_computed():
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
    labels = np.array([0, 1, 1, 0])
    # bins of width 0.5: all confidences in [0.5, 1) -> one bin, conf mean 0.75, acc 0.5
    ece, mce, bins = cal.expected_calibration_error(probs, labels, n_bins=2)
    assert np.isclose(ece, 0.25) and np.isclose(mce, 0.25)
    assert bins["count"].tolist() == [0, 4]


def test_adaptive_ece_equal_mass_bins():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 1000)
    probs = np.stack([conf, 1 - conf], axis=1)
    labels = (rng.random(1000) < conf).astype(int) * 0  # label 0 correct w.p. conf
    labels = np.where(rng.random(1000) < conf, 0, 1)
    assert cal.adaptive_ece(probs, labels, 10) < 0.05  # perfectly calibrated synthetic data


def test_temperature_scaling_fits_known_temperature():
    torch.manual_seed(0)
    N, K = 4000, 5
    logits_true = torch.randn(N, K) * 2
    labels = torch.distributions.Categorical(logits=logits_true).sample()
    over_conf = logits_true * 3.0  # true T = 3 undoes it
    ts = cal.TemperatureScaling().fit(over_conf, labels)
    assert abs(ts.temperature - 3.0) < 0.3
    nll_before = torch.nn.functional.cross_entropy(over_conf, labels)
    nll_after = torch.nn.functional.cross_entropy(ts(over_conf), labels)
    assert nll_after < nll_before
    assert torch.equal(ts(over_conf).argmax(1), over_conf.argmax(1))  # accuracy unchanged
    p_before = torch.softmax(over_conf, 1).numpy(); p_after = torch.softmax(ts(over_conf), 1).detach().numpy()
    assert cal.expected_calibration_error(p_after, labels.numpy())[0] < cal.expected_calibration_error(p_before, labels.numpy())[0]
