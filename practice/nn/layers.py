# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/layers.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k layers -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/layers --force

"""NumPy layers with explicit ``forward`` / ``backward`` (Part III, chapters 1-2).

Conventions (see STYLE.md):

* Row-major data: ``X`` is ``(N, d_in)``, one example per row, so a linear
  layer computes ``Z = X @ W + b`` with ``W`` of shape ``(d_in, d_out)``.
* ``forward`` caches exactly what ``backward`` needs and nothing more.
* ``backward(dout)`` takes ``dL/d(output)`` with the output's shape and returns
  ``dL/d(input)`` with the input's shape; parameter gradients are stored on the
  layer (``self.dW``, ``self.db``) so an optimizer can read them.
"""
from __future__ import annotations
import math
from typing import Callable
import numpy as np

class Layer:
    """Base class: a differentiable map with optional parameters."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

    def params(self) -> list[np.ndarray]:
        """Trainable arrays, in a fixed order matched by :meth:`grads`."""
        raise NotImplementedError('TODO: implement params (see the reference in src/mlbook)')

    def grads(self) -> list[np.ndarray]:
        """Gradients of the loss w.r.t. :meth:`params`, same order."""
        raise NotImplementedError('TODO: implement grads (see the reference in src/mlbook)')

class Linear(Layer):
    """Affine layer ``Z = X W + b``.

    Shapes: ``X`` ``(N, d_in)``, ``W`` ``(d_in, d_out)``, ``b`` ``(d_out,)``,
    ``Z`` ``(N, d_out)``.

    Backward (derived in chapter 2):
        ``dX = dZ W^T``, ``dW = X^T dZ``, ``db = sum_n dZ[n, :]``.
    """

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

    def params(self) -> list[np.ndarray]:
        raise NotImplementedError('TODO: implement params (see the reference in src/mlbook)')

    def grads(self) -> list[np.ndarray]:
        raise NotImplementedError('TODO: implement grads (see the reference in src/mlbook)')

class ReLU(Layer):
    """``h = max(z, 0)``; ``dz = dh * 1[z > 0]``. Any shape."""

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class Sigmoid(Layer):
    """``s = 1 / (1 + exp(-z))``; ``ds/dz = s (1 - s)``. Any shape."""

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class Tanh(Layer):
    """``t = tanh(z)``; ``dt/dz = 1 - t^2``. Any shape."""

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class GELU(Layer):
    """Exact GELU ``g = z Phi(z)`` with ``Phi`` the standard-normal CDF.

    ``dg/dz = Phi(z) + z phi(z)`` where ``phi`` is the standard-normal pdf.
    Uses the erf identity ``Phi(z) = 0.5 (1 + erf(z / sqrt 2))`` with the
    standard library's exact ``math.erf`` vectorised over the array.
    """

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class SiLU(Layer):
    """SiLU / Swish-1: ``s = z sigmoid(z)``; ``ds/dz = sig(z) (1 + z (1 - sig(z)))``."""

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class Softmax(Layer):
    """Row-wise softmax ``p_k = exp(z_k) / sum_j exp(z_j)`` over the last axis.

    Backward (derived in chapter 2, Jacobian ``diag(p) - p p^T``):
        ``dz = p * (dp - sum_k dp_k p_k)``  (the sum is per row).
    Only use this layer when you need probabilities as an *output*; for the
    training loss use ``CrossEntropyLoss`` which fuses log-softmax and NLL.
    """

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')
_erf = np.vectorize(math.erf, otypes=[np.float64])

def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF ``Phi(z) = 0.5 (1 + erf(z / sqrt 2))`` (exact, via ``math.erf``)."""
    raise NotImplementedError('TODO: implement _normal_cdf (see the reference in src/mlbook)')

def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float=1e-06) -> np.ndarray:
    """Central finite-difference gradient of scalar ``f`` at ``x`` (any shape).

    ``g[i] = (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)``. O(numel) evaluations,
    so only for tests on small arrays. ``x`` must be float64 for ~1e-7 accuracy.
    """
    raise NotImplementedError('TODO: implement numerical_gradient (see the reference in src/mlbook)')

def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """``max |a - b| / max(|a| + |b|, 1e-8)`` - the usual gradient-check metric."""
    raise NotImplementedError('TODO: implement rel_error (see the reference in src/mlbook)')
