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
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))  # (N, d_hidden)
        return self.fc2(h)  # (N, K)


def input_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """grad_x of cross-entropy loss. x (N, ...) -> (N, ...)."""
    x = x.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)  # (N, ...)
    return grad


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
         clip: tuple[float, float] | None = (0.0, 1.0)) -> torch.Tensor:
    """One-step L-inf attack. x (N, ...) -> x_adv (N, ...)."""
    grad = input_gradient(model, x, y)  # (N, ...)
    x_adv = x + eps * grad.sign()  # (N, ...)
    if clip is not None:
        x_adv = x_adv.clamp(*clip)
    return x_adv.detach()


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, alpha: float,
        n_steps: int, clip: tuple[float, float] | None = (0.0, 1.0), random_start: bool = True) -> torch.Tensor:
    """Iterative L-inf attack with projection onto the eps-ball around x."""
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)  # (N, ...)
    for _ in range(n_steps):
        grad = input_gradient(model, x_adv, y)  # (N, ...)
        x_adv = x_adv + alpha * grad.sign()  # (N, ...)
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)  # project onto L-inf ball
        if clip is not None:
            x_adv = x_adv.clamp(*clip)
    return x_adv.detach()


def adversarial_training_step(model: nn.Module, opt: torch.optim.Optimizer, x: torch.Tensor,
                              y: torch.Tensor, eps: float, alpha: float, n_steps: int) -> float:
    """Madry-style min-max step: attack with PGD, then train on the adversarial batch."""
    x_adv = pgd(model, x, y, eps, alpha, n_steps)  # (N, ...)
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x_adv), y)
    loss.backward()
    opt.step()
    return float(loss.detach())
