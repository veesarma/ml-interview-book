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

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def params(self) -> list[np.ndarray]:
        """Trainable arrays, in a fixed order matched by :meth:`grads`."""
        return []

    def grads(self) -> list[np.ndarray]:
        """Gradients of the loss w.r.t. :meth:`params`, same order."""
        return []


class Linear(Layer):
    """Affine layer ``Z = X W + b``.

    Shapes: ``X`` ``(N, d_in)``, ``W`` ``(d_in, d_out)``, ``b`` ``(d_out,)``,
    ``Z`` ``(N, d_out)``.

    Backward (derived in chapter 2):
        ``dX = dZ W^T``, ``dW = X^T dZ``, ``db = sum_n dZ[n, :]``.
    """

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator | None = None) -> None:
        rng = np.random.default_rng(0) if rng is None else rng
        # He/Kaiming-normal init (chapter 4): std = sqrt(2 / fan_in).
        self.W = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, d_out))  # (d_in, d_out)
        self.b = np.zeros(d_out)  # (d_out,)
        self.dW = np.zeros_like(self.W)  # (d_in, d_out)
        self.db = np.zeros_like(self.b)  # (d_out,)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x  # (N, d_in) cached for dW
        return x @ self.W + self.b  # (N, d_out); b broadcasts over N

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._x is not None, "call forward first"
        x = self._x  # (N, d_in)
        self.dW = x.T @ dout  # (d_in, N) @ (N, d_out) -> (d_in, d_out)
        self.db = dout.sum(axis=0)  # (d_out,)   sum over the broadcast (batch) axis
        return dout @ self.W.T  # (N, d_out) @ (d_out, d_in) -> (N, d_in)

    def params(self) -> list[np.ndarray]:
        return [self.W, self.b]

    def grads(self) -> list[np.ndarray]:
        return [self.dW, self.db]


class ReLU(Layer):
    """``h = max(z, 0)``; ``dz = dh * 1[z > 0]``. Any shape."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._mask = z > 0  # same shape as z, bool
        return z * self._mask  # same shape as z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._mask is not None
        return dout * self._mask  # same shape as z


class Sigmoid(Layer):
    """``s = 1 / (1 + exp(-z))``; ``ds/dz = s (1 - s)``. Any shape."""

    def __init__(self) -> None:
        self._s: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        # Numerically stable: never exponentiate a large positive number.
        pos = z >= 0  # same shape, bool
        out = np.empty_like(z, dtype=np.float64)  # same shape as z
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        self._s = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._s is not None
        return dout * self._s * (1.0 - self._s)  # same shape as z


class Tanh(Layer):
    """``t = tanh(z)``; ``dt/dz = 1 - t^2``. Any shape."""

    def __init__(self) -> None:
        self._t: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._t = np.tanh(z)  # same shape as z
        return self._t

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._t is not None
        return dout * (1.0 - self._t ** 2)  # same shape as z


class GELU(Layer):
    """Exact GELU ``g = z Phi(z)`` with ``Phi`` the standard-normal CDF.

    ``dg/dz = Phi(z) + z phi(z)`` where ``phi`` is the standard-normal pdf.
    Uses the erf identity ``Phi(z) = 0.5 (1 + erf(z / sqrt 2))`` with the
    standard library's exact ``math.erf`` vectorised over the array.
    """

    def __init__(self) -> None:
        self._z: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._z = z  # same shape as z
        return z * _normal_cdf(z)  # same shape as z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._z is not None
        z = self._z
        pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)  # same shape as z
        return dout * (_normal_cdf(z) + z * pdf)  # same shape as z


class SiLU(Layer):
    """SiLU / Swish-1: ``s = z sigmoid(z)``; ``ds/dz = sig(z) (1 + z (1 - sig(z)))``."""

    def __init__(self) -> None:
        self._z: np.ndarray | None = None
        self._sig: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        self._z = z  # same shape as z
        self._sig = Sigmoid().forward(z)  # same shape as z
        return z * self._sig  # same shape as z

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._z is not None and self._sig is not None
        s = self._sig
        return dout * (s * (1.0 + self._z * (1.0 - s)))  # same shape as z


class Softmax(Layer):
    """Row-wise softmax ``p_k = exp(z_k) / sum_j exp(z_j)`` over the last axis.

    Backward (derived in chapter 2, Jacobian ``diag(p) - p p^T``):
        ``dz = p * (dp - sum_k dp_k p_k)``  (the sum is per row).
    Only use this layer when you need probabilities as an *output*; for the
    training loss use ``CrossEntropyLoss`` which fuses log-softmax and NLL.
    """

    def __init__(self) -> None:
        self._p: np.ndarray | None = None

    def forward(self, z: np.ndarray) -> np.ndarray:
        shifted = z - z.max(axis=-1, keepdims=True)  # (..., K); shift for stability
        e = np.exp(shifted)  # (..., K)
        self._p = e / e.sum(axis=-1, keepdims=True)  # (..., K)
        return self._p

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._p is not None
        p = self._p  # (..., K)
        dot = (dout * p).sum(axis=-1, keepdims=True)  # (..., 1)
        return p * (dout - dot)  # (..., K)


_erf = np.vectorize(math.erf, otypes=[np.float64])


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard-normal CDF ``Phi(z) = 0.5 (1 + erf(z / sqrt 2))`` (exact, via ``math.erf``)."""
    return 0.5 * (1.0 + _erf(z / np.sqrt(2.0)))  # same shape as z


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of scalar ``f`` at ``x`` (any shape).

    ``g[i] = (f(x + eps e_i) - f(x - eps e_i)) / (2 eps)``. O(numel) evaluations,
    so only for tests on small arrays. ``x`` must be float64 for ~1e-7 accuracy.
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)  # same shape as x
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f(x)
        x[idx] = old - eps
        f_minus = f(x)
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """``max |a - b| / max(|a| + |b|, 1e-8)`` - the usual gradient-check metric."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))
