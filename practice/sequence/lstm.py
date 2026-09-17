# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/sequence/lstm.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k lstm -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py sequence/lstm --force

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
    raise NotImplementedError('TODO: implement sigmoid (see the reference in src/mlbook)')

class LSTM:
    """Single-layer LSTM; parameters are one (W_x, W_h, b) triple per gate.

    ``self.params`` keys: ``W_x{f,i,o,g}`` (d_in, d_h), ``W_h{f,i,o,g}`` (d_h, d_h),
    ``b_{f,i,o,g}`` (d_h,). The forget bias is initialised to 1 (Jozefowicz et al.)
    so the carousel starts open.
    """
    GATES = ('f', 'i', 'o', 'g')

    def __init__(self, d_in: int, d_h: int, seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _gates(self, x_t: np.ndarray, h_prev: np.ndarray) -> tuple[np.ndarray, ...]:
        """Compute (f, i, o, g) for one step. x_t: (d_in,), h_prev: (d_h,)."""
        raise NotImplementedError('TODO: implement _gates (see the reference in src/mlbook)')

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Run the recurrence from zero state.

        Args:
            X: (T, d_in) inputs.
        Returns:
            H: (T, d_h) hidden states h_1..h_T.
            cache: everything ``backward`` needs.
        """
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dH: np.ndarray, cache: dict) -> dict[str, np.ndarray]:
        """Full BPTT given dL/dH.

        Args:
            dH: (T, d_h) upstream gradient w.r.t. every hidden state.
        Returns:
            dict of gradients with the same keys/shapes as ``self.params``.
        """
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')
