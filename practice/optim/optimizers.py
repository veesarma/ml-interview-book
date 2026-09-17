# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/optim/optimizers.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k optimizers -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py optim/optimizers --force

"""First-order optimizers from scratch in NumPy with one common interface.

Every optimizer holds a list of parameter arrays and updates them *in place*
from a matching list of gradients::

    opt = Adam([W, b], lr=1e-3)
    ...
    opt.step([dW, db])   # W and b are modified in place

Update rules use the notation of the chapter: ``theta`` parameters, ``g``
gradient, ``eta`` learning rate, ``t`` step count starting at 1.
"""
from __future__ import annotations
import numpy as np

class Optimizer:
    """Base class: stores parameters, exposes ``step(grads)``.

    Args:
        params: list of arrays, each of any shape; updated in place.
        lr: learning rate ``eta``.
    """

    def __init__(self, params: list[np.ndarray], lr: float) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def step(self, grads: list[np.ndarray]) -> None:
        raise NotImplementedError('TODO: implement step (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class SGD(Optimizer):
    """``theta <- theta - eta g`` (optionally with coupled L2 weight decay ``g += wd theta``)."""

    def __init__(self, params: list[np.ndarray], lr: float, weight_decay: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class Momentum(Optimizer):
    """Heavy-ball momentum: ``v <- mu v + g``, ``theta <- theta - eta v`` (PyTorch convention)."""

    def __init__(self, params: list[np.ndarray], lr: float, momentum: float=0.9) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class Nesterov(Optimizer):
    """Nesterov momentum in the PyTorch form: ``v <- mu v + g``, step with ``g + mu v``.

    Equivalent to evaluating the gradient at the look-ahead point
    ``theta + mu v`` under a change of variables.
    """

    def __init__(self, params: list[np.ndarray], lr: float, momentum: float=0.9) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class AdaGrad(Optimizer):
    """``G <- G + g^2``, ``theta <- theta - eta g / (sqrt(G) + eps)``.

    The accumulated square ``G`` only grows, so the effective LR decays to zero.
    """

    def __init__(self, params: list[np.ndarray], lr: float, eps: float=1e-10) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class RMSProp(Optimizer):
    """``s <- rho s + (1-rho) g^2``, ``theta <- theta - eta g / (sqrt(s) + eps)``.

    An exponential moving average replaces AdaGrad's ever-growing sum.
    """

    def __init__(self, params: list[np.ndarray], lr: float, rho: float=0.99, eps: float=1e-08) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction, matching ``torch.optim.Adam``.

    ``m <- b1 m + (1-b1) g``;  ``v <- b2 v + (1-b2) g^2``
    ``m_hat = m / (1 - b1^t)``;  ``v_hat = v / (1 - b2^t)``
    ``theta <- theta - eta m_hat / (sqrt(v_hat) + eps)``

    ``weight_decay`` here is *coupled* L2 (``g <- g + wd theta`` before the
    moments), exactly as in ``torch.optim.Adam``. See ``AdamW`` for the decoupled form.
    """

    def __init__(self, params: list[np.ndarray], lr: float=0.001, betas: tuple[float, float]=(0.9, 0.999), eps: float=1e-08, weight_decay: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

class AdamW(Adam):
    """AdamW (Loshchilov & Hutter, 2019): Adam + *decoupled* weight decay.

    ``theta <- theta - eta (m_hat / (sqrt(v_hat) + eps) + wd theta)``

    The decay is applied directly to the weights and never passes through the
    adaptive denominator, so every weight shrinks at the same rate ``eta wd``
    regardless of its gradient history. Matches ``torch.optim.AdamW``.
    """

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        raise NotImplementedError('TODO: implement _update (see the reference in src/mlbook)')

def run_optimizer(opt: Optimizer, grad_fn, n_steps: int) -> np.ndarray:
    """Run ``opt`` for ``n_steps`` on a single parameter and record the trajectory.

    Args:
        grad_fn: maps the parameter (d,) to its gradient (d,).
    Returns:
        (n_steps + 1, d) trajectory including the starting point.
    """
    raise NotImplementedError('TODO: implement run_optimizer (see the reference in src/mlbook)')
