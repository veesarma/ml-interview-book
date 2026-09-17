# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/mlp.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k mlp -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/mlp --force

"""A complete MLP in NumPy: forward, backward, SGD step, and a synthetic task.

The point of this module is chapter 2 of Part III: everything backprop needs
is ~60 lines once each layer owns its ``forward``/``backward``.
"""
from __future__ import annotations
import numpy as np
from mlbook.nn.layers import GELU, Layer, Linear, ReLU, Sigmoid, Tanh
from mlbook.nn.losses import CrossEntropyLoss
_ACTIVATIONS = {'relu': ReLU, 'gelu': GELU, 'tanh': Tanh, 'sigmoid': Sigmoid}

class MLP:
    """``Linear -> act -> Linear -> act -> ... -> Linear`` (no activation on the logits).

    ``sizes = [d_in, h1, ..., K]``. Forward input ``(N, d_in)``, output logits ``(N, K)``.
    """

    def __init__(self, sizes: list[int], activation: str='relu', seed: int=0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        """Propagate ``dL/dlogits`` ``(N, K)`` back to ``dL/dx`` ``(N, d_in)``.

        Walks the layers in reverse; each layer stores its own parameter grads.
        """
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

    def params(self) -> list[np.ndarray]:
        raise NotImplementedError('TODO: implement params (see the reference in src/mlbook)')

    def grads(self) -> list[np.ndarray]:
        raise NotImplementedError('TODO: implement grads (see the reference in src/mlbook)')

    def sgd_step(self, lr: float) -> None:
        """In-place ``theta <- theta - lr * dL/dtheta`` for every parameter."""
        raise NotImplementedError('TODO: implement sgd_step (see the reference in src/mlbook)')

def make_two_moons(n: int=400, noise: float=0.1, seed: int=0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles. Returns ``X`` ``(n, 2)`` and ``y`` ``(n,)`` in {0, 1}."""
    raise NotImplementedError('TODO: implement make_two_moons (see the reference in src/mlbook)')

def train_classifier(model: MLP, x: np.ndarray, y: np.ndarray, epochs: int=200, lr: float=0.1, batch_size: int=32, seed: int=0) -> list[float]:
    """Minibatch SGD on softmax cross-entropy. Returns the per-epoch mean loss.

    ``x`` ``(N, d_in)``, ``y`` ``(N,)`` int labels.
    """
    raise NotImplementedError('TODO: implement train_classifier (see the reference in src/mlbook)')

def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of rows whose argmax logit equals the label."""
    raise NotImplementedError('TODO: implement accuracy (see the reference in src/mlbook)')
