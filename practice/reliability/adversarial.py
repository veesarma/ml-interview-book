# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/reliability/adversarial.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k adversarial -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py reliability/adversarial --force

"""Adversarial examples: FGSM and PGD (L-inf) in PyTorch, plus a tiny MLP for tests.

FGSM:  x_adv = clip( x + eps * sign(grad_x L(f(x), y)) )
PGD:   x_{t+1} = Proj_{B_eps(x)} ( x_t + alpha * sign(grad_x L(f(x_t), y)) ),
       x_0 = x + Uniform(-eps, eps) (random start).
"""
from __future__ import annotations
import torch
from torch import nn

class TinyMLP(nn.Module):
    """2-layer MLP: (N, d_in) -> (N, K)."""

    def __init__(self, d_in: int, d_hidden: int, n_classes: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def input_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """grad_x of cross-entropy loss. x (N, ...) -> (N, ...)."""
    raise NotImplementedError('TODO: implement input_gradient (see the reference in src/mlbook)')

def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, clip: tuple[float, float] | None=(0.0, 1.0)) -> torch.Tensor:
    """One-step L-inf attack. x (N, ...) -> x_adv (N, ...)."""
    raise NotImplementedError('TODO: implement fgsm (see the reference in src/mlbook)')

def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, n_steps: int, clip: tuple[float, float] | None=(0.0, 1.0), random_start: bool=True) -> torch.Tensor:
    """Iterative L-inf attack with projection onto the eps-ball around x."""
    raise NotImplementedError('TODO: implement pgd (see the reference in src/mlbook)')

def adversarial_training_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float, n_steps: int) -> float:
    """Madry-style min-max step: attack with PGD, then train on the adversarial batch."""
    raise NotImplementedError('TODO: implement adversarial_training_step (see the reference in src/mlbook)')
