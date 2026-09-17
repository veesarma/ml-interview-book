# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/safety/membership_inference.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k membership_inference -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py safety/membership_inference --force

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

def make_data(n: int, d: int=20, seed: int=0) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy binary labels from a random linear rule: X (n, d), y (n,)."""
    raise NotImplementedError('TODO: implement make_data (see the reference in src/mlbook)')

def make_mlp(d: int=20, hidden: int=256, seed: int=0) -> nn.Module:
    raise NotImplementedError('TODO: implement make_mlp (see the reference in src/mlbook)')

def train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float=0.01, weight_decay: float=0.0) -> None:
    raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

@torch.no_grad()
def membership_scores(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """-per-example cross-entropy: (n,). Higher = 'looks like a member'."""
    raise NotImplementedError('TODO: implement membership_scores (see the reference in src/mlbook)')

def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """Mann-Whitney AUC: P(pos > neg) + 0.5 P(pos == neg). pos: (n_p,), neg: (n_n,)."""
    raise NotImplementedError('TODO: implement auc (see the reference in src/mlbook)')

def membership_attack_auc(train_epochs: int, n_train: int=64, n_test: int=256, weight_decay: float=0.0) -> float:
    """Train an MLP on n_train points, score members vs fresh non-members, return AUC."""
    raise NotImplementedError('TODO: implement membership_attack_auc (see the reference in src/mlbook)')
