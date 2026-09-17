# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/sequence/rnn.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k rnn -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py sequence/rnn --force

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

    def __init__(self, d_in: int, d_h: int, d_out: int, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, X: np.ndarray, h0: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, dict]:
        """Run the recurrence over one sequence.

        Args:
            X: (T, d_in) inputs, one time step per row.
            h0: (d_h,) initial hidden state; zeros if None.
        Returns:
            Y: (T, d_out) read-outs y_t = h_t W_y + b_y.
            H: (T, d_h) hidden states h_1..h_T.
            cache: tensors needed by ``backward``.
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dY: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Full BPTT given dL/dY.

        Args:
            dY: (T, d_out) upstream gradient of the loss w.r.t. each read-out y_t.
            cache: from ``forward``.
        Returns:
            dict with the same keys/shapes as ``self.params``.
        """
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

    def hidden_jacobian_norms(self, cache: dict) -> np.ndarray:
        """Spectral norm of dh_T/dh_t for every t (the "product of Jacobians").

        dh_T/dh_t = prod_{s=t+1}^{T} diag(1 - h_s^2) W_h^T   (evaluated right to left)

        Returns:
            (T,) array; entry t is ||dh_T/dh_t||_2. Entry T-1 is 1 (identity).
        """
        raise NotImplementedError('TODO: implement hidden_jacobian_norms (see the reference in src/mlbook)')

def clip_grad_norm(grads: dict[str, np.ndarray], max_norm: float) -> tuple[dict[str, np.ndarray], float]:
    """Global-norm gradient clipping (Pascanu et al., 2013).

    g <- g * max_norm / ||g||  if ||g|| > max_norm, where ||g|| is the norm of the
    concatenation of all gradients.

    Returns:
        (clipped grads, the pre-clip global norm).
    """
    raise NotImplementedError('TODO: implement clip_grad_norm (see the reference in src/mlbook)')

def truncated_bptt_chunks(T: int, k: int) -> list[tuple[int, int]]:
    """Index ranges [(start, end), ...] for truncated BPTT with window ``k``.

    The forward pass carries h across chunks; the backward pass is cut at each
    chunk boundary (gradients do not flow past ``start``). Used by training loops,
    not by the model class.
    """
    raise NotImplementedError('TODO: implement truncated_bptt_chunks (see the reference in src/mlbook)')
