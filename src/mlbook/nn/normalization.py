"""BatchNorm, LayerNorm, RMSNorm and GroupNorm in NumPy, forward *and* backward
(Part III, chapter 5).

All four share one algebraic core: normalise a vector ``x`` of length ``M`` to
``xhat = (x - mu) / sqrt(var + eps)`` and the backward is

    dx = (1 / sqrt(var + eps)) * ( dxhat - mean(dxhat) - xhat * mean(dxhat * xhat) )

with the means taken over the same ``M`` elements the statistics came from.
What differs is *which axis* ``M`` runs over: the batch (BN), the feature
vector (LN), a group of channels x spatial (GN).
"""
from __future__ import annotations

import numpy as np


def _norm_backward(dxhat: np.ndarray, xhat: np.ndarray, inv_std: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    """Shared backward for ``xhat = (x - mu) * inv_std`` with stats over ``axis``.

    ``dxhat``, ``xhat`` have the input shape; ``inv_std`` broadcasts against them.
    """
    m1 = dxhat.mean(axis=axis, keepdims=True)  # mean of dxhat over the stat axis
    m2 = (dxhat * xhat).mean(axis=axis, keepdims=True)  # mean of dxhat*xhat
    return inv_std * (dxhat - m1 - xhat * m2)  # input shape


class BatchNorm1d:
    """BatchNorm over the batch axis of ``(N, D)`` inputs.

    Train: ``mu = mean_n x``, ``var = mean_n (x - mu)^2`` (biased, over the batch),
    ``y = gamma * xhat + beta``; running stats updated with ``momentum``.
    Eval: use the running stats -> a fixed per-feature affine map.
    Backward returns ``dx`` ``(N, D)`` and stores ``dgamma``, ``dbeta`` ``(D,)``.
    """

    def __init__(self, d: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.gamma = np.ones(d)  # (D,)
        self.beta = np.zeros(d)  # (D,)
        self.dgamma = np.zeros(d)  # (D,)
        self.dbeta = np.zeros(d)  # (D,)
        self.running_mean = np.zeros(d)  # (D,)
        self.running_var = np.ones(d)  # (D,)
        self.eps, self.momentum = eps, momentum
        self.training = True
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mu = x.mean(axis=0)  # (D,)
            var = x.var(axis=0)  # (D,)  biased estimator, as in the paper
            n = x.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            # PyTorch stores the *unbiased* variance in running_var.
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / max(n - 1, 1)
        else:
            mu, var = self.running_mean, self.running_var  # (D,), (D,)
        inv_std = 1.0 / np.sqrt(var + self.eps)  # (D,)
        xhat = (x - mu) * inv_std  # (N, D)
        self._cache = (xhat, inv_std)
        return self.gamma * xhat + self.beta  # (N, D)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        xhat, inv_std = self._cache  # (N, D), (D,)
        self.dgamma = (dout * xhat).sum(axis=0)  # (D,)
        self.dbeta = dout.sum(axis=0)  # (D,)
        dxhat = dout * self.gamma  # (N, D)
        if not self.training:
            return dxhat * inv_std  # eval: mu, var are constants
        return _norm_backward(dxhat, xhat, inv_std, axis=0)  # (N, D)


class BatchNorm2d:
    """BatchNorm for ``(N, C, H, W)``: statistics per channel over ``N*H*W``.

    Implemented by moving ``C`` last and flattening to ``(N*H*W, C)``, which is
    exactly what "per-channel batch statistics" means.
    """

    def __init__(self, c: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        self.bn = BatchNorm1d(c, eps=eps, momentum=momentum)

    @property
    def training(self) -> bool:
        return self.bn.training

    @training.setter
    def training(self, value: bool) -> None:
        self.bn.training = value

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        flat = x.transpose(0, 2, 3, 1).reshape(n * h * w, c)  # (N*H*W, C)
        out = self.bn.forward(flat)  # (N*H*W, C)
        return out.reshape(n, h, w, c).transpose(0, 3, 1, 2)  # (N, C, H, W)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h, w = dout.shape
        flat = dout.transpose(0, 2, 3, 1).reshape(n * h * w, c)  # (N*H*W, C)
        dx = self.bn.backward(flat)  # (N*H*W, C)
        return dx.reshape(n, h, w, c).transpose(0, 3, 1, 2)  # (N, C, H, W)


class LayerNorm:
    """LayerNorm over the last axis of ``(..., D)``: one ``mu``/``var`` per row.

    Identical computation at train and eval time - no batch dependence.
    Backward returns ``dx`` with the input shape; ``dgamma``, ``dbeta`` are ``(D,)``.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        self.gamma = np.ones(d)  # (D,)
        self.beta = np.zeros(d)  # (D,)
        self.dgamma = np.zeros(d)  # (D,)
        self.dbeta = np.zeros(d)  # (D,)
        self.eps = eps
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)  # (..., 1)
        var = x.var(axis=-1, keepdims=True)  # (..., 1)
        inv_std = 1.0 / np.sqrt(var + self.eps)  # (..., 1)
        xhat = (x - mu) * inv_std  # (..., D)
        self._cache = (xhat, inv_std)
        return self.gamma * xhat + self.beta  # (..., D)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        xhat, inv_std = self._cache  # (..., D), (..., 1)
        lead = tuple(range(dout.ndim - 1))  # all axes except the last
        self.dgamma = (dout * xhat).sum(axis=lead)  # (D,)
        self.dbeta = dout.sum(axis=lead)  # (D,)
        dxhat = dout * self.gamma  # (..., D)
        return _norm_backward(dxhat, xhat, inv_std, axis=-1)  # (..., D)


class RMSNorm:
    """RMSNorm (Zhang & Sennrich, 2019): ``y = g * x / sqrt(mean(x^2) + eps)``.

    No mean subtraction and no bias: one fewer reduction than LayerNorm and no
    re-centering. Backward: ``dx = (1/r) * (dxhat - xhat * mean(dxhat * xhat))``.
    """

    def __init__(self, d: int, eps: float = 1e-6) -> None:
        self.g = np.ones(d)  # (D,)
        self.dg = np.zeros(d)  # (D,)
        self.eps = eps
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        inv_rms = 1.0 / np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + self.eps)  # (..., 1)
        xhat = x * inv_rms  # (..., D)
        self._cache = (xhat, inv_rms)
        return self.g * xhat  # (..., D)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        xhat, inv_rms = self._cache  # (..., D), (..., 1)
        lead = tuple(range(dout.ndim - 1))
        self.dg = (dout * xhat).sum(axis=lead)  # (D,)
        dxhat = dout * self.g  # (..., D)
        m2 = (dxhat * xhat).mean(axis=-1, keepdims=True)  # (..., 1)
        return inv_rms * (dxhat - xhat * m2)  # (..., D)


class GroupNorm:
    """GroupNorm (Wu & He, 2018) for ``(N, C, H, W)``: stats over ``(C/G, H, W)`` per group.

    ``G = 1`` is LayerNorm over ``(C, H, W)``; ``G = C`` is InstanceNorm.
    ``gamma``, ``beta`` are per channel ``(C,)``.
    """

    def __init__(self, num_groups: int, c: int, eps: float = 1e-5) -> None:
        assert c % num_groups == 0
        self.groups = num_groups
        self.gamma = np.ones(c)  # (C,)
        self.beta = np.zeros(c)  # (C,)
        self.dgamma = np.zeros(c)  # (C,)
        self.dbeta = np.zeros(c)  # (C,)
        self.eps = eps
        self._cache: tuple[np.ndarray, np.ndarray, tuple[int, ...]] | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        xg = x.reshape(n, self.groups, (c // self.groups) * h * w)  # (N, G, M)
        mu = xg.mean(axis=-1, keepdims=True)  # (N, G, 1)
        var = xg.var(axis=-1, keepdims=True)  # (N, G, 1)
        inv_std = 1.0 / np.sqrt(var + self.eps)  # (N, G, 1)
        xhat = ((xg - mu) * inv_std).reshape(n, c, h, w)  # (N, C, H, W)
        self._cache = (xhat, inv_std, x.shape)
        return self.gamma[None, :, None, None] * xhat + self.beta[None, :, None, None]  # (N, C, H, W)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        xhat, inv_std, (n, c, h, w) = self._cache
        self.dgamma = (dout * xhat).sum(axis=(0, 2, 3))  # (C,)
        self.dbeta = dout.sum(axis=(0, 2, 3))  # (C,)
        dxhat = (dout * self.gamma[None, :, None, None]).reshape(n, self.groups, -1)  # (N, G, M)
        xhat_g = xhat.reshape(n, self.groups, -1)  # (N, G, M)
        dx = _norm_backward(dxhat, xhat_g, inv_std, axis=-1)  # (N, G, M)
        return dx.reshape(n, c, h, w)  # (N, C, H, W)
