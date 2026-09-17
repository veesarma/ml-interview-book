"""Dropout, stochastic depth, label smoothing, mixup/CutMix and early stopping
in NumPy (Part III, chapter 6).
"""
from __future__ import annotations

import numpy as np

from mlbook.nn.losses import log_softmax, one_hot


class Dropout:
    """Inverted dropout: train ``y = x * m / (1 - p)`` with ``m ~ Bernoulli(1 - p)``; eval ``y = x``.

    Dividing by ``1 - p`` at train time keeps ``E[y] = x`` so inference needs no
    rescaling. Backward: ``dx = dy * m / (1 - p)`` (same mask). Any shape.
    """

    def __init__(self, p: float = 0.5, seed: int = 0) -> None:
        assert 0.0 <= p < 1.0
        self.p = p
        self.training = True
        self._rng = np.random.default_rng(seed)
        self._scaled_mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self._scaled_mask = None
            return x
        keep = self._rng.random(x.shape) >= self.p  # same shape as x, bool
        self._scaled_mask = keep / (1.0 - self.p)  # same shape: 0 or 1/(1-p)
        return x * self._scaled_mask  # same shape as x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._scaled_mask is None:
            return dout
        return dout * self._scaled_mask  # same shape as x


def drop_path(x: np.ndarray, p: float, rng: np.random.Generator, training: bool = True) -> np.ndarray:
    """Stochastic depth / DropPath on a residual branch output ``x`` ``(N, ...)``.

    Drops the *whole branch* for a random subset of samples (mask shape
    ``(N, 1, ..., 1)``) and rescales the survivors by ``1 / (1 - p)``, the
    inverted-dropout convention used by timm and torchvision.
    """
    if not training or p == 0.0:
        return x
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (N, 1, ..., 1)
    keep = rng.random(mask_shape) >= p  # (N, 1, ..., 1) bool
    return x * keep / (1.0 - p)  # (N, ...)


def label_smoothing_targets(y: np.ndarray, num_classes: int, eps: float) -> np.ndarray:
    """``q = (1 - eps) * onehot(y) + eps / K``. ``y`` ``(N,)`` int -> ``(N, K)``."""
    return (1.0 - eps) * one_hot(y, num_classes) + eps / num_classes  # (N, K)


class LabelSmoothingCrossEntropy:
    """``L = -(1/N) sum_n sum_k q_{nk} log p_{nk}`` with smoothed targets ``q``.

    ``backward`` returns ``(p - q) / N`` ``(N, K)`` - the ``p - y`` rule with a soft ``y``.
    """

    def __init__(self, num_classes: int, eps: float = 0.1) -> None:
        self.k, self.eps = num_classes, eps
        self._p: np.ndarray | None = None
        self._q: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        logp = log_softmax(logits)  # (N, K)
        self._p = np.exp(logp)  # (N, K)
        self._q = label_smoothing_targets(y, self.k, self.eps)  # (N, K)
        return float(-(self._q * logp).sum(axis=-1).mean())

    def backward(self) -> np.ndarray:
        assert self._p is not None and self._q is not None
        return (self._p - self._q) / self._p.shape[0]  # (N, K)


def mixup(x: np.ndarray, y_onehot: np.ndarray, alpha: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """mixup (Zhang et al., 2018): ``x~ = lam x_i + (1-lam) x_j``, same for one-hot ``y``.

    ``lam ~ Beta(alpha, alpha)`` (one draw per batch, as in the reference code);
    ``j`` is a random permutation of the batch. ``x`` ``(N, ...)``, ``y`` ``(N, K)``.
    """
    lam = rng.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = rng.permutation(x.shape[0])  # (N,)
    x_mix = lam * x + (1.0 - lam) * x[perm]  # (N, ...)
    y_mix = lam * y_onehot + (1.0 - lam) * y_onehot[perm]  # (N, K)
    return x_mix, y_mix


def cutmix(x: np.ndarray, y_onehot: np.ndarray, alpha: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """CutMix (Yun et al., 2019) on images ``x`` ``(N, C, H, W)``.

    Paste a box from ``x[perm]`` into ``x``; the label weight is the *pasted
    area fraction*, so ``lam`` is recomputed from the actual box.
    """
    n, _, h, w = x.shape
    lam = rng.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = rng.permutation(n)  # (N,)
    cut_ratio = np.sqrt(1.0 - lam)  # box side fraction so that area ~ 1 - lam
    bh, bw = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = rng.integers(h), rng.integers(w)
    y0, y1 = np.clip(cy - bh // 2, 0, h), np.clip(cy + bh // 2, 0, h)
    x0, x1 = np.clip(cx - bw // 2, 0, w), np.clip(cx + bw // 2, 0, w)
    x_mix = x.copy()  # (N, C, H, W)
    x_mix[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
    lam_actual = 1.0 - (y1 - y0) * (x1 - x0) / (h * w)
    y_mix = lam_actual * y_onehot + (1.0 - lam_actual) * y_onehot[perm]  # (N, K)
    return x_mix, y_mix


class EarlyStopping:
    """Stop when the validation metric has not improved by ``min_delta`` for ``patience`` checks."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best = np.inf
        self.bad_checks = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Record one validation loss; return True if training should stop."""
        if val_loss < self.best - self.min_delta:
            self.best, self.bad_checks = val_loss, 0
        else:
            self.bad_checks += 1
            self.should_stop = self.bad_checks >= self.patience
        return self.should_stop


def sgd_step_weight_decay(w: np.ndarray, grad: np.ndarray, lr: float, wd: float, decoupled: bool = True) -> np.ndarray:
    """One SGD step with weight decay, in the two forms that differ under Adam.

    Coupled (L2 penalty): ``w <- w - lr * (grad + wd * w)``.
    Decoupled (AdamW-style): ``w <- w - lr * grad - lr * wd * w``.
    For plain SGD they coincide; the distinction matters once ``grad`` is
    rescaled by an adaptive optimizer (see Part I, optimization).
    """
    if decoupled:
        return w - lr * grad - lr * wd * w  # same shape as w
    return w - lr * (grad + wd * w)  # same shape as w
