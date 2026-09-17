"""LSTM in NumPy with every gate written out, forward and full BPTT.

Equations (row-major; one sequence, no batch):

    f_t = sigmoid(x_t W_xf + h_{t-1} W_hf + b_f)      forget gate       (d_h,)
    i_t = sigmoid(x_t W_xi + h_{t-1} W_hi + b_i)      input gate        (d_h,)
    o_t = sigmoid(x_t W_xo + h_{t-1} W_ho + b_o)      output gate       (d_h,)
    g_t = tanh   (x_t W_xg + h_{t-1} W_hg + b_g)      cell candidate    (d_h,)
    c_t = f_t * c_{t-1} + i_t * g_t                   cell state        (d_h,)
    h_t = o_t * tanh(c_t)                             hidden state      (d_h,)

The cell path c_{t-1} -> c_t is *additive* and gated by f_t only, so
dc_{t-1} = f_t * dc_t (+ terms through the gates): with f_t near 1 the error
signal is carried unchanged across many steps -- Hochreiter & Schmidhuber's
"constant error carousel".
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function, elementwise."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))), np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


class LSTM:
    """Single-layer LSTM; parameters are one (W_x, W_h, b) triple per gate.

    ``self.params`` keys: ``W_x{f,i,o,g}`` (d_in, d_h), ``W_h{f,i,o,g}`` (d_h, d_h),
    ``b_{f,i,o,g}`` (d_h,). The forget bias is initialised to 1 (Jozefowicz et al.)
    so the carousel starts open.
    """

    GATES = ("f", "i", "o", "g")

    def __init__(self, d_in: int, d_h: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.d_in, self.d_h = d_in, d_h
        self.params: dict[str, np.ndarray] = {}
        for gate in self.GATES:
            self.params[f"W_x{gate}"] = rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_in, d_h))  # (d_in, d_h)
            self.params[f"W_h{gate}"] = rng.normal(0.0, 1.0 / np.sqrt(d_h), size=(d_h, d_h))  # (d_h, d_h)
            self.params[f"b_{gate}"] = np.zeros(d_h)  # (d_h,)
        self.params["b_f"] += 1.0

    def _gates(self, x_t: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, ...]:
        """Compute (f, i, o, g) for one step. x_t: (d_in,), h_prev: (d_h,)."""
        p = self.params
        f = sigmoid(x_t @ p["W_xf"] + h_prev @ p["W_hf"] + p["b_f"])  # (d_h,)
        i = sigmoid(x_t @ p["W_xi"] + h_prev @ p["W_hi"] + p["b_i"])  # (d_h,)
        o = sigmoid(x_t @ p["W_xo"] + h_prev @ p["W_ho"] + p["b_o"])  # (d_h,)
        g = np.tanh(x_t @ p["W_xg"] + h_prev @ p["W_hg"] + p["b_g"])  # (d_h,)
        return f, i, o, g

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run the recurrence from zero state.

        Args:
            X: (T, d_in) inputs.
        Returns:
            H: (T, d_h) hidden states h_1..h_T.
            cache: everything ``backward`` needs.
        """
        T = X.shape[0]
        d_h = self.d_h
        F, I, O, G = (np.zeros((T, d_h)) for _ in range(4))  # each (T, d_h)
        C, H = np.zeros((T, d_h)), np.zeros((T, d_h))  # (T, d_h)
        C_prev, H_prev = np.zeros((T, d_h)), np.zeros((T, d_h))  # (T, d_h)
        h_prev, c_prev = np.zeros(d_h), np.zeros(d_h)  # (d_h,)
        for t in range(T):
            H_prev[t], C_prev[t] = h_prev, c_prev
            f, i, o, g = self._gates(X[t], h_prev)
            c_t = f * c_prev + i * g  # (d_h,) additive cell update
            h_t = o * np.tanh(c_t)  # (d_h,)
            F[t], I[t], O[t], G[t], C[t], H[t] = f, i, o, g, c_t, h_t
            h_prev, c_prev = h_t, c_t
        cache = {"X": X, "F": F, "I": I, "O": O, "G": G, "C": C, "H": H, "C_prev": C_prev, "H_prev": H_prev}
        return H, cache

    def backward(self, dH: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Full BPTT given dL/dH.

        Args:
            dH: (T, d_h) upstream gradient w.r.t. every hidden state.
        Returns:
            dict of gradients with the same keys/shapes as ``self.params``.
        """
        p = self.params
        X, F, I, O, G, C = cache["X"], cache["F"], cache["I"], cache["O"], cache["G"], cache["C"]
        C_prev, H_prev = cache["C_prev"], cache["H_prev"]
        T = X.shape[0]
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        dh_next = np.zeros(self.d_h)  # (d_h,) from h_{t+1}
        dc_next = np.zeros(self.d_h)  # (d_h,) from c_{t+1}
        for t in reversed(range(T)):
            dh = dH[t] + dh_next  # (d_h,) total gradient into h_t
            tanh_c = np.tanh(C[t])  # (d_h,)
            do = dh * tanh_c  # (d_h,)
            dc = dh * O[t] * (1.0 - tanh_c ** 2) + dc_next  # (d_h,) carousel: dc_t gets dc_{t+1}*f_{t+1} via dc_next
            df = dc * C_prev[t]  # (d_h,)
            di = dc * G[t]  # (d_h,)
            dg = dc * I[t]  # (d_h,)
            # through the gate nonlinearities -> pre-activation gradients
            da = {
                "f": df * F[t] * (1.0 - F[t]),  # sigmoid'
                "i": di * I[t] * (1.0 - I[t]),
                "o": do * O[t] * (1.0 - O[t]),
                "g": dg * (1.0 - G[t] ** 2),  # tanh'
            }
            dh_next = np.zeros(self.d_h)  # (d_h,)
            for gate in self.GATES:
                grads[f"W_x{gate}"] += np.outer(X[t], da[gate])  # (d_in, d_h)
                grads[f"W_h{gate}"] += np.outer(H_prev[t], da[gate])  # (d_h, d_h)
                grads[f"b_{gate}"] += da[gate]  # (d_h,)
                dh_next += da[gate] @ p[f"W_h{gate}"].T  # (d_h,)
            dc_next = dc * F[t]  # (d_h,) the constant error carousel
        return grads
