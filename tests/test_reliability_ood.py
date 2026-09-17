import numpy as np
from scipy.special import logsumexp

from mlbook.evaluation.classification_metrics import roc_auc
from mlbook.reliability import ood


def test_max_softmax_and_energy_scores():
    z = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    msp = ood.max_softmax_score(z)
    assert np.isclose(msp[0], np.exp(2) / (np.exp(2) + 2)) and np.isclose(msp[1], 1 / 3)
    assert np.allclose(ood.energy_score(z, T=1.0), logsumexp(z, axis=1))
    assert np.allclose(ood.energy_score(z, T=2.0), 2 * logsumexp(z / 2, axis=1))
    # energy separates scaled logits that max-softmax cannot see
    z_big = np.array([[10.0, 10.0]]); z_small = np.array([[0.0, 0.0]])
    assert np.isclose(ood.max_softmax_score(z_big), ood.max_softmax_score(z_small))
    assert ood.energy_score(z_big) > ood.energy_score(z_small)


def test_mahalanobis_detector_separates_ood_features():
    rng = np.random.default_rng(0)
    means = np.array([[0, 0], [6, 0], [0, 6]], dtype=float)
    y = rng.integers(0, 3, 600)
    F = means[y] + rng.normal(size=(600, 2))
    det = ood.MahalanobisDetector().fit(F, y)
    assert np.allclose(det.means, means, atol=0.3)
    F_ood = rng.normal(size=(200, 2)) + np.array([15.0, 15.0])
    s_in, s_out = det.score(F), det.score(F_ood)
    labels = np.r_[np.ones(600), np.zeros(200)]
    assert roc_auc(labels, np.r_[s_in, s_out]) > 0.99
