# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/safety/backdoor_demo.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k backdoor_demo -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py safety/backdoor_demo --force

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

def make_clean_data(n: int, seed: int=0) -> tuple[torch.Tensor, torch.Tensor]:
    """X (n, 1, 8, 8) in [0, 1], y (n,) in {0, 1}."""
    raise NotImplementedError('TODO: implement make_clean_data (see the reference in src/mlbook)')

def stamp_trigger(X: torch.Tensor) -> torch.Tensor:
    """White 2x2 patch in the bottom-right corner. (n, 1, 8, 8) -> (n, 1, 8, 8)."""
    raise NotImplementedError('TODO: implement stamp_trigger (see the reference in src/mlbook)')

def poison(X: torch.Tensor, y: torch.Tensor, frac: float, target: int=1, seed: int=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Poison ``frac`` of the non-target examples: add trigger, relabel to ``target``.

    Returns (X_poisoned, y_poisoned, is_poisoned mask (n,) bool).
    """
    raise NotImplementedError('TODO: implement poison (see the reference in src/mlbook)')

class TinyClassifier(nn.Module):
    """(B, 1, 8, 8) -> features (B, 16) -> logits (B, 2)."""

    def __init__(self):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement features (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int=60, lr: float=0.01) -> None:
    raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

@torch.no_grad()
def accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    raise NotImplementedError('TODO: implement accuracy (see the reference in src/mlbook)')

@torch.no_grad()
def attack_success_rate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, target: int=1) -> float:
    """Fraction of non-target clean inputs classified as ``target`` once the trigger is stamped."""
    raise NotImplementedError('TODO: implement attack_success_rate (see the reference in src/mlbook)')

@torch.no_grad()
def spectral_signature_scores(model: TinyClassifier, X: torch.Tensor, y: torch.Tensor, target: int=1) -> tuple[torch.Tensor, torch.Tensor]:
    """Outlier score per target-class example: |<R_i - mean, top right singular vector>|.

    Returns (scores (n_target,), indices into X (n_target,)).
    """
    raise NotImplementedError('TODO: implement spectral_signature_scores (see the reference in src/mlbook)')
