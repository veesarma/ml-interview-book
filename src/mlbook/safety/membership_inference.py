"""Membership inference via loss thresholding (Yeom et al. 2018) and its AUC.

Score s(x, y) = -loss(f(x), y). Members (training examples) tend to have lower loss
than non-members drawn from the same distribution when the model overfits, so the
attacker ranks by score. AUC = P(score_member > score_nonmember) (Mann-Whitney), 0.5
means the attack is no better than chance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def make_data(n: int, d: int = 20, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy binary labels from a random linear rule: X (n, d), y (n,)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)  # (n, d)
    w = torch.randn(d, generator=g)  # (d,)
    logits = X @ w / d**0.5 + 0.8 * torch.randn(n, generator=g)  # (n,) label noise
    return X, (logits > 0).long()


def make_mlp(d: int = 20, hidden: int = 256, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float = 1e-2, weight_decay: float = 0.0) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def membership_scores(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """-per-example cross-entropy: (n,). Higher = 'looks like a member'."""
    return -F.cross_entropy(model(X), y, reduction="none")  # (n,)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5 P(pos == neg). pos: (n_p,), neg: (n_n,)."""
    diff = pos.view(-1, 1) - neg.view(1, -1)  # (n_p, n_n)
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def membership_attack_auc(train_epochs: int, n_train: int = 64, n_test: int = 256, weight_decay: float = 0.0) -> float:
    """Train an MLP on n_train points, score members vs fresh non-members, return AUC."""
    X, y = make_data(n_train + n_test)
    Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]  # (n_train, d), ...
    model = make_mlp()
    train(model, Xtr, ytr, train_epochs, weight_decay=weight_decay)
    return auc(membership_scores(model, Xtr, ytr), membership_scores(model, Xte, yte))
