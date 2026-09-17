import numpy as np
from scipy import linalg

from mlbook.evaluation import generative_metrics as gm


def test_frechet_distance_matches_scipy_sqrtm_reference():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6))
    S1 = A @ A.T + np.eye(6)
    Bm = rng.normal(size=(6, 6))
    S2 = Bm @ Bm.T + np.eye(6)
    mu1, mu2 = rng.normal(size=6), rng.normal(size=6)
    ref = np.sum((mu1 - mu2) ** 2) + np.trace(S1 + S2 - 2 * linalg.sqrtm(S1 @ S2).real)
    assert np.isclose(gm.frechet_distance(mu1, S1, mu2, S2), ref, rtol=1e-6)
    assert np.isclose(gm.frechet_distance(mu1, S1, mu1, S1), 0.0, atol=1e-8)


def test_fid_from_features_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(500, 4))
    assert gm.fid_from_features(F, F) < 1e-8
    assert gm.fid_from_features(F, F + 3.0) > 30


def test_inception_score_bounds():
    K = 5
    confident_diverse = np.eye(K)  # each image a different class, confidently
    assert np.isclose(gm.inception_score(confident_diverse), K)
    uniform = np.full((10, K), 1 / K)
    assert np.isclose(gm.inception_score(uniform), 1.0)


def test_clip_score():
    v = np.array([[1.0, 0.0], [0.0, 1.0]])
    t = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.isclose(gm.clip_score(v, t, w=2.5), 2.5 * (1 + 0) / 2)
