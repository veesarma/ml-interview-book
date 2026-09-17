"""A BadNets-style backdoor (Gu et al. 2017) on a tiny image classifier, plus the
spectral-signature defence (Tran et al. 2018).

Data: 8x8 grey images; class 1 iff the left half is brighter than the right half.
Poisoning: stamp a 2x2 white patch (the trigger) in the bottom-right corner of a
fraction of class-0 images and *relabel them as class 1*. The model learns the
shortcut "patch => class 1" and keeps clean accuracy.

Spectral signature: within the target class, poisoned examples share a direction in
representation space; project onto the top singular vector of the centred features
and flag the largest scores.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def make_clean_data(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """X (n, 1, 8, 8) in [0, 1], y (n,) in {0, 1}."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 1, 8, 8, generator=g)  # (n, 1, 8, 8)
    y = (X[:, 0, :, :4].mean(dim=(1, 2)) > X[:, 0, :, 4:].mean(dim=(1, 2))).long()  # (n,)
    return X, y


def stamp_trigger(X: torch.Tensor) -> torch.Tensor:
    """White 2x2 patch in the bottom-right corner. (n, 1, 8, 8) -> (n, 1, 8, 8)."""
    X = X.clone()
    X[:, 0, 6:8, 6:8] = 1.0
    return X


def poison(X: torch.Tensor, y: torch.Tensor, frac: float, target: int = 1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poison ``frac`` of the non-target examples: add trigger, relabel to ``target``.

    Returns (X_poisoned, y_poisoned, is_poisoned mask (n,) bool).
    """
    g = torch.Generator().manual_seed(seed)
    candidates = torch.nonzero(y != target).squeeze(1)  # (n_nontarget,)
    k = int(frac * X.shape[0])
    chosen = candidates[torch.randperm(len(candidates), generator=g)[:k]]  # (k,)
    Xp, yp = X.clone(), y.clone()
    Xp[chosen] = stamp_trigger(X[chosen])
    yp[chosen] = target
    mask = torch.zeros(X.shape[0], dtype=torch.bool)
    mask[chosen] = True
    return Xp, yp, mask


class TinyClassifier(nn.Module):
    """(B, 1, 8, 8) -> features (B, 16) -> logits (B, 2)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc1(x.flatten(1)))  # (B, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.features(x))  # (B, 2)


def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int = 60, lr: float = 1e-2) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    return float((model(X).argmax(1) == y).float().mean())


@torch.no_grad()
def attack_success_rate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, target: int = 1) -> float:
    """Fraction of non-target clean inputs classified as ``target`` once the trigger is stamped."""
    src = X[y != target]  # (n_src, 1, 8, 8)
    return float((model(stamp_trigger(src)).argmax(1) == target).float().mean())


@torch.no_grad()
def spectral_signature_scores(model: TinyClassifier, X: torch.Tensor, y: torch.Tensor, target: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Outlier score per target-class example: |<R_i - mean, top right singular vector>|.

    Returns (scores (n_target,), indices into X (n_target,)).
    """
    idx = torch.nonzero(y == target).squeeze(1)  # (n_target,)
    R = model.features(X[idx])  # (n_target, 16)
    R = R - R.mean(dim=0, keepdim=True)  # centred
    _, _, Vh = torch.linalg.svd(R, full_matrices=False)  # Vh: (16, 16)
    v = Vh[0]  # (16,) top right-singular vector
    return (R @ v).abs(), idx  # (n_target,), (n_target,)
