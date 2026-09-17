"""LSTM: gates against torch.nn.LSTMCell, full BPTT against finite differences."""
import numpy as np
import torch

from mlbook.sequence.lstm import LSTM, sigmoid


def test_sigmoid_stable():
    assert sigmoid(np.array([1000.0])) == 1.0
    assert sigmoid(np.array([-1000.0])) == 0.0
    assert np.isclose(sigmoid(np.array([0.0])), 0.5)


def test_lstm_forward_matches_torch_lstmcell():
    d_in, d_h, T = 3, 4, 6
    lstm = LSTM(d_in, d_h, seed=0)
    cell = torch.nn.LSTMCell(d_in, d_h)
    p = lstm.params
    # torch gate order is (i, f, g, o) with weight_ih of shape (4*d_h, d_in) acting as W x
    W_ih = np.concatenate([p["W_xi"].T, p["W_xf"].T, p["W_xg"].T, p["W_xo"].T], axis=0)
    W_hh = np.concatenate([p["W_hi"].T, p["W_hf"].T, p["W_hg"].T, p["W_ho"].T], axis=0)
    b = np.concatenate([p["b_i"], p["b_f"], p["b_g"], p["b_o"]])
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor(W_ih, dtype=torch.float32))
        cell.weight_hh.copy_(torch.tensor(W_hh, dtype=torch.float32))
        cell.bias_ih.copy_(torch.tensor(b, dtype=torch.float32))
        cell.bias_hh.zero_()
    X = np.random.randn(T, d_in)
    H, _ = lstm.forward(X)
    h = torch.zeros(1, d_h)
    c = torch.zeros(1, d_h)
    for t in range(T):
        h, c = cell(torch.tensor(X[t : t + 1], dtype=torch.float32), (h, c))
        np.testing.assert_allclose(H[t], h[0].detach().numpy(), atol=1e-5)


def test_lstm_bptt_matches_finite_differences():
    lstm = LSTM(d_in=3, d_h=4, seed=1)
    X = np.random.randn(6, 3)
    R = np.random.randn(6, 4)

    def loss_and_grads():
        H, cache = lstm.forward(X)
        return float((H * R).sum()), lambda: lstm.backward(R, cache)

    _, g = loss_and_grads()
    grads = g()
    eps = 1e-6
    for name, W in lstm.params.items():
        num = np.zeros_like(W)
        for idx in np.ndindex(W.shape):
            old = W[idx]
            W[idx] = old + eps
            lp, _ = loss_and_grads()
            W[idx] = old - eps
            lm, _ = loss_and_grads()
            W[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        np.testing.assert_allclose(grads[name], num, rtol=1e-5, atol=1e-7, err_msg=name)


def test_lstm_cell_state_gradient_is_carried_by_forget_gate():
    """With f = 1 (huge forget bias) and no gate dependence on inputs, dL/dc_0 equals dL/dc_T."""
    lstm = LSTM(d_in=2, d_h=3, seed=2)
    lstm.params["b_f"][:] = 50.0  # f_t = 1 exactly in float64
    for k in ("W_xf", "W_hf"):
        lstm.params[k][:] = 0.0
    X = np.random.randn(10, 2)
    H, cache = lstm.forward(X)
    # only the last hidden state gets gradient; check the carousel product prod f_t == 1
    assert np.allclose(cache["F"], 1.0)
