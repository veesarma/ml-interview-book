"""A complete MLP in NumPy: forward, backward, SGD step, and a synthetic task.

The point of this module is chapter 2 of Part III: everything backprop needs
is ~60 lines once each layer owns its ``forward``/``backward``.
"""
from __future__ import annotations

import numpy as np

from mlbook.nn.layers import GELU, Layer, Linear, ReLU, Sigmoid, Tanh
from mlbook.nn.losses import CrossEntropyLoss

_ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "tanh": Tanh, "sigmoid": Sigmoid}


class MLP:
    """``Linear -> act -> Linear -> act -> ... -> Linear`` (no activation on the logits).

    ``sizes = [d_in, h1, ..., K]``. Forward input ``(N, d_in)``, output logits ``(N, K)``.
    """

    def __init__(self, sizes: list[int], activation: str = "relu", seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        act = _ACTIVATIONS[activation]
        self.layers: list[Layer] = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng=rng))
            if i < len(sizes) - 2:  # no nonlinearity after the last Linear
                self.layers.append(act())

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x  # (N, d_in)
        for layer in self.layers:
            h = layer.forward(h)  # (N, d_layer)
        return h  # (N, K) logits

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        """Propagate ``dL/dlogits`` ``(N, K)`` back to ``dL/dx`` ``(N, d_in)``.

        Walks the layers in reverse; each layer stores its own parameter grads.
        """
        d = dlogits  # (N, K)
        for layer in reversed(self.layers):
            d = layer.backward(d)  # (N, d_layer_in)
        return d  # (N, d_in)

    def params(self) -> list[np.ndarray]:
        return [p for layer in self.layers for p in layer.params()]

    def grads(self) -> list[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads()]

    def sgd_step(self, lr: float) -> None:
        """In-place ``theta <- theta - lr * dL/dtheta`` for every parameter."""
        for p, g in zip(self.params(), self.grads()):
            p -= lr * g


def make_two_moons(n: int = 400, noise: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two interleaving half-circles. Returns ``X`` ``(n, 2)`` and ``y`` ``(n,)`` in {0, 1}."""
    rng = np.random.default_rng(seed)
    n0 = n // 2
    n1 = n - n0
    t0 = np.linspace(0.0, np.pi, n0)  # (n0,)
    t1 = np.linspace(0.0, np.pi, n1)  # (n1,)
    x0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)  # (n0, 2)   upper moon
    x1 = np.stack([1.0 - np.cos(t1), 1.0 - np.sin(t1) - 0.5], axis=1)  # (n1, 2)  lower moon
    x = np.concatenate([x0, x1], axis=0)  # (n, 2)
    x = x + rng.normal(0.0, noise, size=x.shape)  # (n, 2)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])  # (n,)
    perm = rng.permutation(n)
    return x[perm], y[perm]


def train_classifier(
    model: MLP,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    lr: float = 0.1,
    batch_size: int = 32,
    seed: int = 0,
) -> list[float]:
    """Minibatch SGD on softmax cross-entropy. Returns the per-epoch mean loss.

    ``x`` ``(N, d_in)``, ``y`` ``(N,)`` int labels.
    """
    rng = np.random.default_rng(seed)
    loss_fn = CrossEntropyLoss()
    n = x.shape[0]
    history: list[float] = []
    for _ in range(epochs):
        perm = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]  # (B,)
            logits = model.forward(x[idx])  # (B, K)
            loss = loss_fn.forward(logits, y[idx])
            model.backward(loss_fn.backward())  # dlogits is (B, K)
            model.sgd_step(lr)
            epoch_loss += loss * idx.shape[0]
        history.append(epoch_loss / n)
    return history


def accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction of rows whose argmax logit equals the label."""
    pred = model.forward(x).argmax(axis=-1)  # (N,)
    return float((pred == y).mean())
