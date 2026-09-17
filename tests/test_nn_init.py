"""Initialisers: empirical variances match the derivations."""
import numpy as np

from mlbook.nn.init import (
    activation_std_by_depth,
    gpt2_residual_normal,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    orthogonal,
    xavier_normal,
    xavier_uniform,
)


def test_variances_match_formulas():
    rng = np.random.default_rng(0)
    fan_in, fan_out = 400, 300
    assert abs(xavier_uniform(fan_in, fan_out, rng).var() - 2 / (fan_in + fan_out)) < 2e-4
    assert abs(xavier_normal(fan_in, fan_out, rng).var() - 2 / (fan_in + fan_out)) < 2e-4
    assert abs(kaiming_normal(fan_in, fan_out, rng).var() - 2 / fan_in) < 3e-4
    assert abs(kaiming_uniform(fan_in, fan_out, rng).var() - 2 / fan_in) < 3e-4
    assert abs(lecun_normal(fan_in, fan_out, rng).var() - 1 / fan_in) < 2e-4
    w = gpt2_residual_normal(fan_in, fan_out, rng, n_layers=12)
    assert abs(w.std() - 0.02 / np.sqrt(24)) < 2e-4


def test_orthogonal_is_orthogonal():
    rng = np.random.default_rng(0)
    w = orthogonal(6, 6, rng)
    np.testing.assert_allclose(w.T @ w, np.eye(6), atol=1e-10)
    w_wide = orthogonal(3, 8, rng)  # rows orthonormal
    np.testing.assert_allclose(w_wide @ w_wide.T, np.eye(3), atol=1e-10)
    w_tall = orthogonal(8, 3, rng)  # columns orthonormal
    np.testing.assert_allclose(w_tall.T @ w_tall, np.eye(3), atol=1e-10)


def test_forward_variance_preserved_through_depth():
    relu = lambda z: np.maximum(z, 0.0)
    ident = lambda z: z
    # He + ReLU: pre-activation variance stays ~1 => post-ReLU std ~ sqrt(1/2)... use pre-act check
    stds_he = activation_std_by_depth(kaiming_normal, relu, width=256, depth=20)
    assert 0.3 < stds_he[-1] < 1.5  # neither vanished nor exploded
    # LeCun + linear: exactly variance-preserving in expectation
    stds_lecun = activation_std_by_depth(lecun_normal, ident, width=256, depth=20)
    assert 0.5 < stds_lecun[-1] < 2.0
    # LeCun (1/fan_in) + ReLU halves the variance per layer => collapses
    stds_bad = activation_std_by_depth(lecun_normal, relu, width=256, depth=20)
    assert stds_bad[-1] < 0.01
