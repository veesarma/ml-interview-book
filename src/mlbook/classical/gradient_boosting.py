"""Gradient-boosted trees with Newton (second-order) leaf weights, XGBoost style.

Round ``m`` fits a tree ``f_m`` to the second-order Taylor expansion of the loss
around the current prediction ``F_{m-1}``:

    L ≈ Σ_i [ g_i f(x_i) + ½ h_i f(x_i)² ] + γ T + ½ λ Σ_leaves w_j²

with ``g_i = ∂ℓ/∂F``, ``h_i = ∂²ℓ/∂F²``. For a fixed tree structure the optimal
leaf weight and the resulting objective are

    w_j* = -G_j / (H_j + λ),      obj* = -½ Σ_j G_j² / (H_j + λ) + γ T,

so the gain of a split is ``½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ)] - γ``.

Losses: ``"squared"`` (g = F - y, h = 1) and ``"logistic"`` (g = σ(F) - y,
h = σ(F)(1 - σ(F))).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logistic_regression import sigmoid


@dataclass
class _NewtonNode:
    feature: int = -1
    threshold: float = 0.0
    left: "_NewtonNode | None" = None
    right: "_NewtonNode | None" = None
    weight: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None


def leaf_weight(G: float, H: float, lam: float) -> float:
    """``w* = -G / (H + λ)``: the Newton step for one leaf."""
    return -G / (H + lam)


def split_gain(G_L: float, H_L: float, G_R: float, H_R: float, lam: float, gamma: float) -> float:
    """``½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ``."""
    G, H = G_L + G_R, H_L + H_R
    return 0.5 * (G_L**2 / (H_L + lam) + G_R**2 / (H_R + lam) - G**2 / (H + lam)) - gamma


class NewtonTree:
    """A regression tree grown on (g, h) pairs with the XGBoost gain and leaf weights.

    ``fit(X (N, d), g (N,), h (N,))``; ``predict(X (N, d)) -> (N,)`` leaf weights.
    """

    def __init__(self, max_depth: int = 3, lam: float = 1.0, gamma: float = 0.0, min_child_weight: float = 1.0) -> None:
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.min_child_weight = min_child_weight  # minimum Σ h in a leaf (≈ min samples)
        self.root: _NewtonNode | None = None

    def _best_split(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> tuple[int, float, float] | None:
        n, d = X.shape
        G, H = g.sum(), h.sum()
        best: tuple[int, float, float] | None = None
        for j in range(d):
            order = np.argsort(X[:, j], kind="stable")  # (n,)
            xs, gs, hs = X[order, j], g[order], h[order]  # (n,), (n,), (n,)
            G_L = np.cumsum(gs)[:-1]  # (n-1,) prefix sums: left child gets first i examples
            H_L = np.cumsum(hs)[:-1]  # (n-1,)
            G_R, H_R = G - G_L, H - H_L  # (n-1,), (n-1,)
            valid = (xs[:-1] != xs[1:]) & (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)  # (n-1,)
            if not valid.any():
                continue
            gains = 0.5 * (G_L**2 / (H_L + self.lam) + G_R**2 / (H_R + self.lam) - G**2 / (H + self.lam)) - self.gamma  # (n-1,)
            gains = np.where(valid, gains, -np.inf)
            i = int(gains.argmax())
            if best is None or gains[i] > best[2]:
                best = (j, float(0.5 * (xs[i] + xs[i + 1])), float(gains[i]))
        return best

    def _grow(self, X: np.ndarray, g: np.ndarray, h: np.ndarray, depth: int) -> _NewtonNode:
        node = _NewtonNode(weight=leaf_weight(g.sum(), h.sum(), self.lam))
        if depth >= self.max_depth or len(g) < 2:
            return node
        split = self._best_split(X, g, h)
        if split is None or split[2] <= 0.0:
            return node
        j, t, _ = split
        mask = X[:, j] <= t  # (n,) bool
        node.feature, node.threshold = j, t
        node.left = self._grow(X[mask], g[mask], h[mask], depth + 1)
        node.right = self._grow(X[~mask], g[~mask], h[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, g: np.ndarray, h: np.ndarray) -> "NewtonTree":
        self.root = self._grow(X, g, h, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(X.shape[0])  # (N,)
        for i, x in enumerate(X):
            node = self.root
            while not node.is_leaf:  # type: ignore[union-attr]
                node = node.left if x[node.feature] <= node.threshold else node.right  # type: ignore[union-attr]
            out[i] = node.weight  # type: ignore[union-attr]
        return out


class GradientBoostedTrees:
    """Newton boosting: ``F_m = F_{m-1} + η f_m`` with ``f_m`` a :class:`NewtonTree`.

    ``loss`` in {"squared", "logistic"}. ``fit(X (N, d), y (N,))``;
    ``predict_raw(X) -> (N,)`` returns ``F_M``; ``predict`` thresholds for logistic.
    ``train_loss_`` records the loss after every round (monotone for small η).
    """

    def __init__(
        self, n_rounds: int = 100, learning_rate: float = 0.1, max_depth: int = 3,
        lam: float = 1.0, gamma: float = 0.0, loss: str = "squared", subsample: float = 1.0, seed: int = 0,
    ) -> None:
        self.n_rounds = n_rounds
        self.lr = learning_rate
        self.max_depth = max_depth
        self.lam = lam
        self.gamma = gamma
        self.loss = loss
        self.subsample = subsample
        self.rng = np.random.default_rng(seed)
        self.trees: list[NewtonTree] = []
        self.base_score = 0.0
        self.train_loss_: list[float] = []

    def _grad_hess(self, F: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First and second derivatives of the per-example loss w.r.t. ``F``. Both (N,)."""
        if self.loss == "squared":
            return F - y, np.ones_like(F)
        p = sigmoid(F)  # (N,)
        return p - y, p * (1.0 - p)

    def _loss_value(self, F: np.ndarray, y: np.ndarray) -> float:
        if self.loss == "squared":
            return float(0.5 * ((F - y) ** 2).mean())
        return float(np.mean(np.logaddexp(0.0, F) - y * F))  # -[y log σ(F) + (1-y) log(1-σ(F))]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedTrees":
        N = X.shape[0]
        if self.loss == "squared":
            self.base_score = float(y.mean())
        else:
            p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.base_score = float(np.log(p0 / (1 - p0)))  # log-odds of the prior
        F = np.full(N, self.base_score)  # (N,) current prediction
        self.trees, self.train_loss_ = [], []
        for _ in range(self.n_rounds):
            g, h = self._grad_hess(F, y)  # (N,), (N,)
            idx = np.arange(N)
            if self.subsample < 1.0:
                idx = self.rng.choice(N, size=int(self.subsample * N), replace=False)  # (N_sub,)
            tree = NewtonTree(self.max_depth, self.lam, self.gamma).fit(X[idx], g[idx], h[idx])
            F = F + self.lr * tree.predict(X)  # (N,)
            self.trees.append(tree)
            self.train_loss_.append(self._loss_value(F, y))
        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """``F_M(x) = base + η Σ_m f_m(x)``. ``X``: (N, d) -> (N,)."""
        F = np.full(X.shape[0], self.base_score)  # (N,)
        for tree in self.trees:
            F = F + self.lr * tree.predict(X)  # (N,)
        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``σ(F_M(x))`` for the logistic loss. ``X``: (N, d) -> (N,)."""
        return sigmoid(self.predict_raw(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.loss == "logistic":
            return (self.predict_raw(X) >= 0.0).astype(int)  # (N,)
        return self.predict_raw(X)
