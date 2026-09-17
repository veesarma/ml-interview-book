"""Gradient magnitude vs. distance back in time: vanilla RNN (product of Jacobians) for three
spectral radii of W_h, and an LSTM cell-state path (constant error carousel).

Writes docs/assets/figures/part05_vanishing_gradients.png. Run from the repo root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.sequence.rnn import VanillaRNN  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_vanishing_gradients.png"


def lstm_cell_gradient_norms(T: int, d_in: int, d_h: int, forget_bias: float) -> np.ndarray:
    """||dL/dc_t|| for L = <h_T, r>, via autograd on a torch LSTMCell with the forget bias set."""
    torch.manual_seed(0)
    cell = torch.nn.LSTMCell(d_in, d_h)
    with torch.no_grad():
        cell.bias_ih[d_h : 2 * d_h] = forget_bias  # torch gate order: i, f, g, o
        cell.bias_hh[d_h : 2 * d_h] = 0.0
    x = torch.randn(T, 1, d_in)
    h, c = torch.zeros(1, d_h), torch.zeros(1, d_h)
    cs = []
    for t in range(T):
        h, c = cell(x[t], (h, c))
        c.retain_grad()
        cs.append(c)
    (h * torch.randn(1, d_h)).sum().backward()
    return np.array([float(ct.grad.norm()) for ct in cs])


def main() -> None:
    T, d_in, d_h = 60, 4, 32
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor="white")
    distance = np.arange(T)[::-1]  # steps back from the last time step
    for scale, label in ((0.6, r"RNN, $\rho(W_h)\approx0.6$ (vanishes)"), (1.0, r"RNN, $\rho(W_h)\approx1.0$"), (1.6, r"RNN, $\rho(W_h)\approx1.6$ (explodes)")):
        rnn = VanillaRNN(d_in, d_h, 1, seed=0)
        W = rnn.params["W_h"]
        rnn.params["W_h"] = W * (scale / np.max(np.abs(np.linalg.eigvals(W))))  # set the spectral radius
        X = np.random.default_rng(0).normal(size=(T, d_in)) * 0.3
        _, _, cache = rnn.forward(X)
        norms = rnn.hidden_jacobian_norms(cache)
        ax.semilogy(distance, norms, label=label)
    for fb, label in ((1.0, "LSTM cell path, forget bias 1"), (3.0, "LSTM cell path, forget bias 3")):
        g = lstm_cell_gradient_norms(T, d_in, d_h, fb)
        ax.semilogy(distance, g / g[-1], linestyle="--", label=label)
    ax.set_xlabel("steps back from the last time step  (T - t)")
    ax.set_ylabel(r"$\|\partial h_T / \partial h_t\|_2$  (LSTM: $\|\partial L/\partial c_t\|$, normalised)")
    ax.set_title("Why gradients vanish or explode through a recurrence", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(1e-12, 1e12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
