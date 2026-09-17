"""Vanilla (Elman) RNN in NumPy with forward pass and full backpropagation through time.

Equations (row-major, one sequence of length T, no batch dimension):

    h_t = tanh(x_t W_x + h_{t-1} W_h + b)          h_t in R^{d_h}
    y_t = h_t W_y + b_y                            y_t in R^{d_out}

BPTT:  with a_t = x_t W_x + h_{t-1} W_h + b and an upstream gradient dh_t
(which sums the gradient coming from y_t and the one flowing back from h_{t+1}),

    da_t   = dh_t * (1 - h_t^2)                    (tanh')
    dW_x  += x_t^T da_t,   dW_h += h_{t-1}^T da_t,   db += da_t
    dh_{t-1} (from the recurrence) = da_t W_h^T

so the gradient reaching h_0 from h_T is a product of T Jacobians
diag(1 - h_t^2) W_h^T, whose norm shrinks or grows geometrically: the vanishing /
exploding gradient problem.
"""

from __future__ import annotations

import numpy as np


class VanillaRNN:
    """Single-layer tanh RNN with a linear read-out.

    Parameters are stored in ``self.params`` with shapes
        W_x: (d_in, d_h)   W_h: (d_h, d_h)   b: (d_h,)
        W_y: (d_h, d_out)  b_y: (d_out,)
    """

    def __init__(self, d_in: int, d_h: int, d_out: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.d_in, self.d_h, self.d_out = d_in, d_h, d_out
        self.params: dict[str, np.ndarray] = {
            "W_x": rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(d_in, d_h)),  # (d_in, d_h)
            "W_h": rng.normal(0.0, 1.0 / np.sqrt(d_h), size=(d_h, d_h)),  # (d_h, d_h)
            "b": np.zeros(d_h),  # (d_h,)
            "W_y": rng.normal(0.0, 1.0 / np.sqrt(d_h), size=(d_h, d_out)),  # (d_h, d_out)
            "b_y": np.zeros(d_out),  # (d_out,)
        }

    def forward(self, X: np.ndarray, h0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run the recurrence over one sequence.

        Args:
            X: (T, d_in) inputs, one time step per row.
            h0: (d_h,) initial hidden state; zeros if None.
        Returns:
            Y: (T, d_out) read-outs y_t = h_t W_y + b_y.
            H: (T, d_h) hidden states h_1..h_T.
            cache: tensors needed by ``backward``.
        """
        p = self.params
        T = X.shape[0]
        h_prev = np.zeros(self.d_h) if h0 is None else h0  # (d_h,)
        H = np.zeros((T, self.d_h))  # (T, d_h)
        H_prev = np.zeros((T, self.d_h))  # (T, d_h): h_{t-1} for each t
        for t in range(T):
            H_prev[t] = h_prev
            a_t = X[t] @ p["W_x"] + h_prev @ p["W_h"] + p["b"]  # (d_h,) pre-activation
            h_prev = np.tanh(a_t)  # (d_h,)
            H[t] = h_prev
        Y = H @ p["W_y"] + p["b_y"]  # (T, d_h) @ (d_h, d_out) -> (T, d_out)
        cache = {"X": X, "H": H, "H_prev": H_prev}
        return Y, H, cache

    def backward(self, dY: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Full BPTT given dL/dY.

        Args:
            dY: (T, d_out) upstream gradient of the loss w.r.t. each read-out y_t.
            cache: from ``forward``.
        Returns:
            dict with the same keys/shapes as ``self.params``.
        """
        p = self.params
        X, H, H_prev = cache["X"], cache["H"], cache["H_prev"]
        T = X.shape[0]
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        grads["W_y"] = H.T @ dY  # (d_h, T) @ (T, d_out) -> (d_h, d_out)
        grads["b_y"] = dY.sum(axis=0)  # (d_out,)
        dH_from_y = dY @ p["W_y"].T  # (T, d_out) @ (d_out, d_h) -> (T, d_h)
        dh_next = np.zeros(self.d_h)  # (d_h,) gradient flowing back from h_{t+1}
        for t in reversed(range(T)):
            dh_t = dH_from_y[t] + dh_next  # (d_h,) total gradient into h_t
            da_t = dh_t * (1.0 - H[t] ** 2)  # (d_h,) through tanh
            grads["W_x"] += np.outer(X[t], da_t)  # (d_in, d_h)
            grads["W_h"] += np.outer(H_prev[t], da_t)  # (d_h, d_h)
            grads["b"] += da_t  # (d_h,)
            dh_next = da_t @ p["W_h"].T  # (d_h,) @ (d_h, d_h) -> (d_h,)
        return grads

    def hidden_jacobian_norms(self, cache: dict) -> np.ndarray:
        """Spectral norm of dh_T/dh_t for every t (the "product of Jacobians").

        dh_T/dh_t = prod_{s=t+1}^{T} diag(1 - h_s^2) W_h^T   (evaluated right to left)

        Returns:
            (T,) array; entry t is ||dh_T/dh_t||_2. Entry T-1 is 1 (identity).
        """
        H = cache["H"]
        T = H.shape[0]
        norms = np.zeros(T)
        J = np.eye(self.d_h)  # (d_h, d_h) running product, starts as dh_T/dh_T
        norms[T - 1] = 1.0
        for t in reversed(range(T - 1)):
            # one more step back: dh_{t+1}/dh_t = diag(1 - h_{t+1}^2) W_h^T
            J = J @ (np.diag(1.0 - H[t + 1] ** 2) @ self.params["W_h"].T)  # (d_h, d_h)
            norms[t] = np.linalg.norm(J, ord=2)
        return norms


def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float) -> tuple[dict[str, np.ndarray], float]:
    """Global-norm gradient clipping (Pascanu et al., 2013).

    g <- g * max_norm / ||g||  if ||g|| > max_norm, where ||g|| is the norm of the
    concatenation of all gradients.

    Returns:
        (clipped grads, the pre-clip global norm).
    """
    total = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())))
    if total <= max_norm or total == 0.0:
        return grads, total
    scale = max_norm / total
    return {k: g * scale for k, g in grads.items()}, total


def truncated_bptt_chunks(T: int, k: int) -> list[tuple[int, int]]:
    """Index ranges [(start, end), ...] for truncated BPTT with window ``k``.

    The forward pass carries h across chunks; the backward pass is cut at each
    chunk boundary (gradients do not flow past ``start``). Used by training loops,
    not by the model class.
    """
    return [(s, min(s + k, T)) for s in range(0, T, k)]
