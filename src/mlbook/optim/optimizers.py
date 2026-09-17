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
        self.params = params
        self.lr = lr
        self.t = 0  # step counter

    def step(self, grads: list[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            p -= self._update(i, p, g)  # in-place: theta <- theta - update

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class SGD(Optimizer):
    """``theta <- theta - eta g`` (optionally with coupled L2 weight decay ``g += wd theta``)."""

    def __init__(self, params: list[np.ndarray], lr: float, weight_decay: float = 0.0) -> None:
        super().__init__(params, lr)
        self.wd = weight_decay

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        return self.lr * (g + self.wd * p)  # same shape as p


class Momentum(Optimizer):
    """Heavy-ball momentum: ``v <- mu v + g``, ``theta <- theta - eta v`` (PyTorch convention)."""

    def __init__(self, params: list[np.ndarray], lr: float, momentum: float = 0.9) -> None:
        super().__init__(params, lr)
        self.mu = momentum
        self.v = [np.zeros_like(p) for p in params]  # velocity, one per param

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.v[i] = self.mu * self.v[i] + g  # same shape as p
        return self.lr * self.v[i]


class Nesterov(Optimizer):
    """Nesterov momentum in the PyTorch form: ``v <- mu v + g``, step with ``g + mu v``.

    Equivalent to evaluating the gradient at the look-ahead point
    ``theta + mu v`` under a change of variables.
    """

    def __init__(self, params: list[np.ndarray], lr: float, momentum: float = 0.9) -> None:
        super().__init__(params, lr)
        self.mu = momentum
        self.v = [np.zeros_like(p) for p in params]

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.v[i] = self.mu * self.v[i] + g  # same shape as p
        return self.lr * (g + self.mu * self.v[i])


class AdaGrad(Optimizer):
    """``G <- G + g^2``, ``theta <- theta - eta g / (sqrt(G) + eps)``.

    The accumulated square ``G`` only grows, so the effective LR decays to zero.
    """

    def __init__(self, params: list[np.ndarray], lr: float, eps: float = 1e-10) -> None:
        super().__init__(params, lr)
        self.eps = eps
        self.G = [np.zeros_like(p) for p in params]  # sum of squared grads

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.G[i] += g * g  # same shape as p
        return self.lr * g / (np.sqrt(self.G[i]) + self.eps)


class RMSProp(Optimizer):
    """``s <- rho s + (1-rho) g^2``, ``theta <- theta - eta g / (sqrt(s) + eps)``.

    An exponential moving average replaces AdaGrad's ever-growing sum.
    """

    def __init__(self, params: list[np.ndarray], lr: float, rho: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__(params, lr)
        self.rho, self.eps = rho, eps
        self.s = [np.zeros_like(p) for p in params]  # EMA of g^2

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.s[i] = self.rho * self.s[i] + (1 - self.rho) * g * g  # same shape as p
        return self.lr * g / (np.sqrt(self.s[i]) + self.eps)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction, matching ``torch.optim.Adam``.

    ``m <- b1 m + (1-b1) g``;  ``v <- b2 v + (1-b2) g^2``
    ``m_hat = m / (1 - b1^t)``;  ``v_hat = v / (1 - b2^t)``
    ``theta <- theta - eta m_hat / (sqrt(v_hat) + eps)``

    ``weight_decay`` here is *coupled* L2 (``g <- g + wd theta`` before the
    moments), exactly as in ``torch.optim.Adam``. See ``AdamW`` for the decoupled form.
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.b1, self.b2 = betas
        self.eps, self.wd = eps, weight_decay
        self.m = [np.zeros_like(p) for p in params]  # first moment
        self.v = [np.zeros_like(p) for p in params]  # second moment

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        g = g + self.wd * p  # coupled L2: enters the moment estimates
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g  # same shape as p
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g  # same shape as p
        m_hat = self.m[i] / (1 - self.b1**self.t)  # bias-corrected
        v_hat = self.v[i] / (1 - self.b2**self.t)  # bias-corrected
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW (Loshchilov & Hutter, 2019): Adam + *decoupled* weight decay.

    ``theta <- theta - eta (m_hat / (sqrt(v_hat) + eps) + wd theta)``

    The decay is applied directly to the weights and never passes through the
    adaptive denominator, so every weight shrinks at the same rate ``eta wd``
    regardless of its gradient history. Matches ``torch.optim.AdamW``.
    """

    def _update(self, i: int, p: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g  # same shape as p
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g  # same shape as p
        m_hat = self.m[i] / (1 - self.b1**self.t)
        v_hat = self.v[i] / (1 - self.b2**self.t)
        return self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.wd * p)


def run_optimizer(opt: Optimizer, grad_fn, n_steps: int) -> np.ndarray:
    """Run ``opt`` for ``n_steps`` on a single parameter and record the trajectory.

    Args:
        grad_fn: maps the parameter (d,) to its gradient (d,).
    Returns:
        (n_steps + 1, d) trajectory including the starting point.
    """
    traj = [opt.params[0].copy()]
    for _ in range(n_steps):
        opt.step([grad_fn(opt.params[0])])
        traj.append(opt.params[0].copy())
    return np.stack(traj)  # (n_steps + 1, d)
