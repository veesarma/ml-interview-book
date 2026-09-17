"""Vanilla RNN: forward against a hand-rolled loop, full BPTT against finite differences."""
import numpy as np

from mlbook.sequence.rnn import VanillaRNN, clip_grad_norm, truncated_bptt_chunks


def _loss_and_grads(rnn: VanillaRNN, X: np.ndarray, R: np.ndarray):
    Y, _, cache = rnn.forward(X)
    loss = float((Y * R).sum())  # linear read-out so dL/dY = R exactly
    return loss, rnn.backward(R, cache)


def test_vanilla_rnn_forward_matches_manual_recurrence():
    rnn = VanillaRNN(d_in=3, d_h=4, d_out=2, seed=1)
    X = np.random.randn(5, 3)
    Y, H, _ = rnn.forward(X)
    p = rnn.params
    h = np.zeros(4)
    for t in range(5):
        h = np.tanh(X[t] @ p["W_x"] + h @ p["W_h"] + p["b"])
        np.testing.assert_allclose(H[t], h, atol=1e-12)
        np.testing.assert_allclose(Y[t], h @ p["W_y"] + p["b_y"], atol=1e-12)


def test_vanilla_rnn_bptt_matches_finite_differences():
    rnn = VanillaRNN(d_in=3, d_h=5, d_out=2, seed=2)
    X = np.random.randn(7, 3)
    R = np.random.randn(7, 2)
    _, grads = _loss_and_grads(rnn, X, R)
    eps = 1e-6
    for name, W in rnn.params.items():
        num = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + eps
            lp, _ = _loss_and_grads(rnn, X, R)
            W[idx] = old - eps
            lm, _ = _loss_and_grads(rnn, X, R)
            W[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(grads[name], num, rtol=1e-5, atol=1e-7, err_msg=name)


def test_hidden_jacobian_norms_shrink_for_small_weights():
    rnn = VanillaRNN(d_in=2, d_h=8, d_out=1, seed=3)
    rnn.params["W_h"] *= 0.3  # spectral radius well below 1 -> vanishing
    X = np.random.randn(30, 2)
    _, _, cache = rnn.forward(X)
    norms = rnn.hidden_jacobian_norms(cache)
    assert norms[-1] == 1.0
    assert norms[0] < 1e-6 < norms[-5]


def test_clip_grad_norm_scales_to_max_norm():
    grads = {"a": np.full(4, 3.0), "b": np.full(4, 4.0)}  # global norm = sqrt(16*9 + 16*16) = 20
    clipped, norm = clip_grad_norm(grads, max_norm=5.0)
    assert np.isclose(norm, 20.0)
    total = np.sqrt(sum((g ** 2).sum() for g in clipped.values()))
    assert np.isclose(total, 5.0)
    same, _ = clip_grad_norm(grads, max_norm=100.0)
    assert same is grads


def test_truncated_bptt_chunks_cover_sequence():
    assert truncated_bptt_chunks(10, 4) == [(0, 4), (4, 8), (8, 10)]
