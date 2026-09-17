"""A tensor-valued reverse-mode autograd engine in NumPy (Part III, chapter 3).

Design, in one paragraph: a ``Tensor`` wraps a NumPy array, remembers which
``Tensor``s produced it (``_prev``) and a closure ``_backward`` that, given
``self.grad``, *accumulates* into the parents' ``.grad``. ``backward()`` sorts
the graph topologically, seeds ``grad = 1`` at the root and runs each closure
in reverse topological order - exactly what PyTorch's engine does, minus the
C++, the multithreading and the in-place bookkeeping.

Every op follows the same template::

    out = Tensor(f(a.data, b.data), (a, b))
    def _backward():
        a.grad += unbroadcast(local_a * out.grad, a.shape)
        b.grad += unbroadcast(local_b * out.grad, b.shape)
    out._backward = _backward
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` (the inverse of NumPy broadcasting).

    Broadcasting *copies* a value across new/size-1 axes in the forward pass;
    the chain rule therefore *sums* the incoming gradient over those axes.
    """
    # 1. axes that broadcasting prepended (grad.ndim > len(shape))
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)  # drop one leading axis
    # 2. axes that were size 1 in the input and got stretched
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)  # keep the 1
    return grad  # shape == shape


class Tensor:
    """Minimal differentiable tensor. ``data`` and ``grad`` are float64 ndarrays."""

    def __init__(self, data, requires_grad: bool = False, _children: Iterable["Tensor"] = (), _op: str = "") -> None:
        self.data = np.asarray(data, dtype=np.float64)  # any shape
        self.grad = np.zeros_like(self.data)  # same shape as data
        self.requires_grad = requires_grad
        self._prev: tuple[Tensor, ...] = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    # ------------------------------------------------------------------ utils
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r})"

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def backward(self) -> None:
        """Reverse-mode AD from this (scalar) tensor to every ancestor."""
        assert self.data.size == 1, "backward() needs a scalar root; call .sum() first"
        order = _topological_order(self)
        self.grad = np.ones_like(self.data)  # dL/dL = 1
        for node in reversed(order):  # children before parents
            node._backward()

    # --------------------------------------------------------------- arithmetic
    def __add__(self, other) -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="add")  # broadcast shape

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.shape)
            other.grad += unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = _as_tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op="mul")  # broadcast shape

        def _backward() -> None:
            self.grad += unbroadcast(other.data * out.grad, self.shape)
            other.grad += unbroadcast(self.data * out.grad, other.shape)

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Tensor":
        assert isinstance(exponent, (int, float)), "only scalar exponents"
        out = Tensor(self.data ** exponent, _children=(self,), _op="pow")  # same shape

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-_as_tensor(other))

    def __truediv__(self, other) -> "Tensor":
        return self * (_as_tensor(other) ** -1.0)

    def __radd__(self, other) -> "Tensor":
        return self + other

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __rsub__(self, other) -> "Tensor":
        return _as_tensor(other) - self

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """``C = A @ B``; ``dA = dC @ B^T``, ``dB = A^T @ dC`` (batched via swapaxes)."""
        out = Tensor(self.data @ other.data, _children=(self, other), _op="matmul")  # (..., n, p)

        def _backward() -> None:
            bt = np.swapaxes(other.data, -1, -2)  # (..., p, m)
            at = np.swapaxes(self.data, -1, -2)  # (..., m, n)
            self.grad += unbroadcast(out.grad @ bt, self.shape)  # (..., n, m)
            other.grad += unbroadcast(at @ out.grad, other.shape)  # (..., m, p)

        out._backward = _backward
        return out

    # --------------------------------------------------------------- reductions
    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), _children=(self,), _op="sum")

        def _backward() -> None:
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis)  # restore the reduced axes as size 1
            self.grad += np.broadcast_to(g, self.shape)  # copy the gradient to every summed element

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        count = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(count))

    # ------------------------------------------------------------ elementwise
    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), _children=(self,), _op="exp")  # same shape

        def _backward() -> None:
            self.grad += out.data * out.grad  # d exp = exp

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), _children=(self,), _op="log")  # same shape

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), _children=(self,), _op="relu")  # same shape

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))  # same shape
        out = Tensor(s, _children=(self,), _op="sigmoid")

        def _backward() -> None:
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)  # same shape
        out = Tensor(t, _children=(self,), _op="tanh")

        def _backward() -> None:
            self.grad += (1.0 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """``log p = z - logsumexp(z)``; ``dz = dlogp - p * sum(dlogp)`` along ``axis``."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)  # (..., K)
        logp = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))  # (..., K)
        out = Tensor(logp, _children=(self,), _op="log_softmax")

        def _backward() -> None:
            p = np.exp(logp)  # (..., K)
            self.grad += out.grad - p * out.grad.sum(axis=axis, keepdims=True)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """``p = softmax(z)``; ``dz = p * (dp - sum(dp * p))`` along ``axis``."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)  # (..., K)
        e = np.exp(shifted)  # (..., K)
        p = e / e.sum(axis=axis, keepdims=True)  # (..., K)
        out = Tensor(p, _children=(self,), _op="softmax")

        def _backward() -> None:
            self.grad += p * (out.grad - (out.grad * p).sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    # ------------------------------------------------------------ shape ops
    def reshape(self, *shape: int) -> "Tensor":
        out = Tensor(self.data.reshape(*shape), _children=(self,), _op="reshape")  # shape

        def _backward() -> None:
            self.grad += out.grad.reshape(self.shape)  # gradient is just reshaped back

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        axes_t = tuple(axes) if axes else tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(axes_t), _children=(self,), _op="transpose")  # permuted shape

        def _backward() -> None:
            inverse = np.argsort(axes_t)  # undo the permutation
            self.grad += out.grad.transpose(tuple(inverse))

        out._backward = _backward
        return out

    @property
    def T(self) -> "Tensor":
        return self.transpose()


def _as_tensor(x) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _topological_order(root: Tensor) -> list[Tensor]:
    """Iterative DFS post-order: every node appears after all of its inputs."""
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))
    return order


def cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy built from engine ops. ``logits`` ``(N, K)``, ``y`` ``(N,)`` int."""
    n, k = logits.shape
    onehot = np.zeros((n, k))  # (N, K)
    onehot[np.arange(n), y] = 1.0
    logp = logits.log_softmax(axis=-1)  # (N, K)
    return -(logp * Tensor(onehot)).sum() * (1.0 / n)  # scalar
