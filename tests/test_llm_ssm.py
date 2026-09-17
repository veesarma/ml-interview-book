"""Tests for the SSM toys (Part VI, chapter 3)."""

import torch

from mlbook.llm.ssm import (
    SelectiveSSM,
    discretize_zoh,
    selective_scan,
    ssm_convolutional,
    ssm_kernel,
    ssm_recurrent,
)


def test_zoh_matches_exact_solution_for_scalar_system():
    # h' = a h + b u with constant u=1 from h(0)=0 has h(Δ) = (exp(aΔ)-1)/a · b
    A, B = torch.tensor([-0.5]), torch.tensor([2.0])
    A_bar, B_bar = discretize_zoh(A, B, delta=0.1)
    assert torch.isclose(A_bar, torch.exp(torch.tensor(-0.05)))
    assert torch.isclose(B_bar, (torch.exp(torch.tensor(-0.05)) - 1) / -0.5 * 2.0)


def test_recurrent_equals_convolutional():
    torch.manual_seed(0)
    N, T = 6, 40
    A = -torch.rand(N) - 0.1
    B, C = torch.randn(N), torch.randn(N)
    A_bar, B_bar = discretize_zoh(A, B, delta=0.2)
    x = torch.randn(T)
    y_rec = ssm_recurrent(x, A_bar, B_bar, C, D=0.3)
    K = ssm_kernel(A_bar, B_bar, C, L=T)
    y_conv = ssm_convolutional(x, K, D=0.3)
    assert torch.allclose(y_rec, y_conv, atol=1e-5)


def test_selective_scan_reduces_to_lti_when_inputs_are_constant():
    torch.manual_seed(0)
    N, T, d = 4, 12, 1
    A = (-torch.rand(N) - 0.1)[None, :]  # (d=1, N)
    B_vec, C_vec = torch.randn(N), torch.randn(N)
    delta = torch.full((1, T, d), 0.3)
    x = torch.randn(1, T, d)
    y_sel = selective_scan(x, delta, A, B_vec[None, None].expand(1, T, N), C_vec[None, None].expand(1, T, N), torch.tensor([0.0]))
    # Mamba's discretisation B̄ = Δ B; build the same LTI system by hand
    A_bar = torch.exp(0.3 * A[0])
    y_lti = ssm_recurrent(x[0, :, 0], A_bar, 0.3 * B_vec, C_vec, D=0.0)
    assert torch.allclose(y_sel[0, :, 0], y_lti, atol=1e-5)


def test_selective_ssm_layer_shapes_and_causality():
    layer = SelectiveSSM(d_model=8, d_state=4)
    x = torch.randn(2, 10, 8)
    y = layer(x)
    assert y.shape == (2, 10, 8)
    y_prefix = layer(x[:, :5])
    assert torch.allclose(y[:, :5], y_prefix, atol=1e-6)
