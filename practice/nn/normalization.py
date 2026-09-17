# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/nn/normalization.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k normalization -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py nn/normalization --force

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
    raise NotImplementedError('TODO: implement _norm_backward (see the reference in src/mlbook)')

class BatchNorm1d:
    """BatchNorm over the batch axis of ``(N, D)`` inputs.

    Train: ``mu = mean_n x``, ``var = mean_n (x - mu)^2`` (biased, over the batch),
    ``y = gamma * xhat + beta``; running stats updated with ``momentum``.
    Eval: use the running stats -> a fixed per-feature affine map.
    Backward returns ``dx`` ``(N, D)`` and stores ``dgamma``, ``dbeta`` ``(D,)``.
    """

    def __init__(self, d: int, eps: float=1e-05, momentum: float=0.1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class BatchNorm2d:
    """BatchNorm for ``(N, C, H, W)``: statistics per channel over ``N*H*W``.

    Implemented by moving ``C`` last and flattening to ``(N*H*W, C)``, which is
    exactly what "per-channel batch statistics" means.
    """

    def __init__(self, c: int, eps: float=1e-05, momentum: float=0.1) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @property
    def training(self) -> bool:
        raise NotImplementedError('TODO: implement training (see the reference in src/mlbook)')

    @training.setter
    def training(self, value: bool) -> None:
        raise NotImplementedError('TODO: implement training (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class LayerNorm:
    """LayerNorm over the last axis of ``(..., D)``: one ``mu``/``var`` per row.

    Identical computation at train and eval time - no batch dependence.
    Backward returns ``dx`` with the input shape; ``dgamma``, ``dbeta`` are ``(D,)``.
    """

    def __init__(self, d: int, eps: float=1e-05) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class RMSNorm:
    """RMSNorm (Zhang & Sennrich, 2019): ``y = g * x / sqrt(mean(x^2) + eps)``.

    No mean subtraction and no bias: one fewer reduction than LayerNorm and no
    re-centering. Backward: ``dx = (1/r) * (dxhat - xhat * mean(dxhat * xhat))``.
    """

    def __init__(self, d: int, eps: float=1e-06) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')

class GroupNorm:
    """GroupNorm (Wu & He, 2018) for ``(N, C, H, W)``: stats over ``(C/G, H, W)`` per group.

    ``G = 1`` is LayerNorm over ``(C, H, W)``; ``G = C`` is InstanceNorm.
    ``gamma``, ``beta`` are per channel ``(C,)``.
    """

    def __init__(self, num_groups: int, c: int, eps: float=1e-05) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement backward (see the reference in src/mlbook)')
