"""Tests for scaling_laws (Part VI, chapter 2)."""

import numpy as np

from mlbook.llm.scaling_laws import (
    CHINCHILLA_FIT,
    ChinchillaParams,
    chinchilla_loss,
    compute_optimal,
    effective_data_with_repetition,
    fit_power_law,
    fit_scaling_law,
    loss_at_fixed_compute,
    tokens_per_parameter,
    training_flops,
)


def test_compute_optimal_is_the_argmin_on_the_isoflop_curve():
    C = 1e22
    N_star, D_star = compute_optimal(C)
    assert abs(training_flops(N_star, D_star) - C) / C < 1e-9
    N_grid = np.logspace(np.log10(N_star) - 1, np.log10(N_star) + 1, 2001)
    losses = loss_at_fixed_compute(C, N_grid)
    N_best = N_grid[np.argmin(losses)]
    assert abs(np.log(N_best) - np.log(N_star)) < 0.01


def test_tokens_per_parameter_is_about_twenty():
    assert 15 < tokens_per_parameter(1e21) < 30
    assert 15 < tokens_per_parameter(1e24) < 30


def test_chinchilla_point_estimate():
    # Hoffmann et al.: Chinchilla is ~70B params on ~1.4T tokens at Gopher's budget (~5.76e23 FLOPs).
    N_star, D_star = compute_optimal(5.76e23)
    assert 4e10 < N_star < 1.2e11
    assert 8e11 < D_star < 2.5e12


def test_fit_scaling_law_recovers_parameters_from_small_runs():
    truth = ChinchillaParams(E=1.7, A=400.0, B=400.0, alpha=0.3, beta=0.3)
    rng = np.random.default_rng(0)
    N = np.array([1e7, 3e7, 1e8, 3e8, 1e9, 1e7, 1e8, 1e9, 3e7, 3e8])
    D = np.array([2e8, 6e8, 2e9, 6e9, 2e10, 2e9, 2e10, 2e11, 6e9, 6e10])
    L = chinchilla_loss(N, D, truth) + rng.normal(0, 1e-3, size=N.shape)
    fit = fit_scaling_law(N, D, L, alpha_grid=np.linspace(0.2, 0.4, 21), beta_grid=np.linspace(0.2, 0.4, 21))
    assert abs(fit.alpha - truth.alpha) < 0.03 and abs(fit.beta - truth.beta) < 0.03
    assert abs(fit.E - truth.E) < 0.05
    pred = chinchilla_loss(np.array([5e9]), np.array([1e11]), fit)
    assert abs(pred - chinchilla_loss(np.array([5e9]), np.array([1e11]), truth)) < 0.02


def test_fit_power_law():
    x = np.logspace(6, 10, 9)
    c, k = fit_power_law(x, 3.0 * x**-0.076)
    assert abs(c - 3.0) < 1e-6 and abs(k - 0.076) < 1e-6


def test_effective_data_saturates_with_repetition():
    U = 1e9
    assert effective_data_with_repetition(0.5e9, U) == 0.5e9
    four_epochs = effective_data_with_repetition(4 * U, U)
    assert 0.9 * 4 * U < four_epochs < 4 * U  # ~4 epochs are almost as good as unique data
    hundred = effective_data_with_repetition(100 * U, U)
    assert hundred < 17 * U  # value of repeats decays to zero: D' -> U (1 + R*)
    assert chinchilla_loss(1e9, 1e10, CHINCHILLA_FIT) > CHINCHILLA_FIT.E
