# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/autograd.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k autograd -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/autograd --force

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
    raise NotImplementedError('TODO: implement unbroadcast (see the reference in src/mlbook)')

class Tensor:
    """Minimal differentiable tensor. ``data`` and ``grad`` are float64 ndarrays."""

    def __init__(self, data, requires_grad: bool=False, _children: Iterable['Tensor']=(), _op: str='') -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError('TODO: implement shape (see the reference in src/mlbook)')

    def __repr__(self) -> str:
        raise NotImplementedError('TODO: implement __repr__ (see the reference in src/mlbook)')

    def zero_grad(self) -> None:
        raise NotImplementedError('TODO: implement zero_grad (see the reference in src/mlbook)')

    def backward(self) -> None:
        """Reverse-mode AD from this (scalar) tensor to every ancestor."""
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

    def __add__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __add__ (see the reference in src/mlbook)')

    def __mul__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __mul__ (see the reference in src/mlbook)')

    def __pow__(self, exponent: float) -> 'Tensor':
        raise NotImplementedError('TODO: implement __pow__ (see the reference in src/mlbook)')

    def __neg__(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement __neg__ (see the reference in src/mlbook)')

    def __sub__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __sub__ (see the reference in src/mlbook)')

    def __truediv__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __truediv__ (see the reference in src/mlbook)')

    def __radd__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __radd__ (see the reference in src/mlbook)')

    def __rmul__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __rmul__ (see the reference in src/mlbook)')

    def __rsub__(self, other) -> 'Tensor':
        raise NotImplementedError('TODO: implement __rsub__ (see the reference in src/mlbook)')

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """``C = A @ B``; ``dA = dC @ B^T``, ``dB = A^T @ dC`` (batched via swapaxes)."""
        raise NotImplementedError('TODO: implement __matmul__ (see the reference in src/mlbook)')

    def sum(self, axis: int | tuple[int, ...] | None=None, keepdims: bool=False) -> 'Tensor':
        raise NotImplementedError('TODO: implement sum (see the reference in src/mlbook)')

    def mean(self, axis: int | tuple[int, ...] | None=None, keepdims: bool=False) -> 'Tensor':
        raise NotImplementedError('TODO: implement mean (see the reference in src/mlbook)')

    def exp(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement exp (see the reference in src/mlbook)')

    def log(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement log (see the reference in src/mlbook)')

    def relu(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement relu (see the reference in src/mlbook)')

    def sigmoid(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement sigmoid (see the reference in src/mlbook)')

    def tanh(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement tanh (see the reference in src/mlbook)')

    def log_softmax(self, axis: int=-1) -> 'Tensor':
        """``log p = z - logsumexp(z)``; ``dz = dlogp - p * sum(dlogp)`` along ``axis``."""
        raise NotImplementedError('TODO: implement log_softmax (see the reference in src/mlbook)')

    def softmax(self, axis: int=-1) -> 'Tensor':
        """``p = softmax(z)``; ``dz = p * (dp - sum(dp * p))`` along ``axis``."""
        raise NotImplementedError('TODO: implement softmax (see the reference in src/mlbook)')

    def reshape(self, *shape: int) -> 'Tensor':
        raise NotImplementedError('TODO: implement reshape (see the reference in src/mlbook)')

    def transpose(self, *axes: int) -> 'Tensor':
        raise NotImplementedError('TODO: implement transpose (see the reference in src/mlbook)')

    @property
    def T(self) -> 'Tensor':
        raise NotImplementedError('TODO: implement T (see the reference in src/mlbook)')

def _as_tensor(x) -> Tensor:
    raise NotImplementedError('TODO: implement _as_tensor (see the reference in src/mlbook)')

def _topological_order(root: Tensor) -> list[Tensor]:
    """Iterative DFS post-order: every node appears after all of its inputs."""
    raise NotImplementedError('TODO: implement _topological_order (see the reference in src/mlbook)')

def cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy built from engine ops. ``logits`` ``(N, K)``, ``y`` ``(N,)`` int."""
    raise NotImplementedError('TODO: implement cross_entropy (see the reference in src/mlbook)')
