"""LoRA: low-rank adaptation of a frozen linear layer.

    h = x W^T + (α / r) · x A^T B^T,     A ∈ R^{r×in},  B ∈ R^{out×r},  r ≪ min(in, out)

B is initialised to zero so the adapted model equals the base model at step 0; A is
Gaussian/Kaiming so the gradient to B is non-zero.  Merging W' = W + (α/r) B A gives
a plain Linear with no inference overhead; unmerging subtracts it back.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen ``nn.Linear`` plus a trainable rank-``r`` update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)  # W and b are frozen
        in_f, out_f = base.in_features, base.out_features
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))  # (r, in)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))  # (out, r)  zero => ΔW = 0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        """ΔW = (α/r) · B A, shape (out, in)."""
        return self.scaling * (self.lora_B @ self.lora_A)  # (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, out)."""
        y = self.base(x)  # (…, out) frozen path (already includes ΔW when merged)
        if self.merged:
            return y
        low_rank = self.dropout(x) @ self.lora_A.T  # (…, r)   project down
        return y + self.scaling * (low_rank @ self.lora_B.T)  # (…, out) project up

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight for zero-overhead inference."""
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo ``merge`` so training can continue on the separate A, B."""
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False


def apply_lora(model: nn.Module, r: int, alpha: float, target_names: tuple[str, ...] = ("q_proj", "v_proj")) -> list[str]:
    """Replace every ``nn.Linear`` whose attribute name is in ``target_names`` with a LoRALinear.

    Returns the dotted names that were wrapped.  Then call ``mark_only_lora_trainable``.
    """
    wrapped: list[str] = []
    for parent_name, parent in list(model.named_modules()):
        for attr, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and attr in target_names:
                setattr(parent, attr, LoRALinear(child, r, alpha))
                wrapped.append(f"{parent_name}.{attr}" if parent_name else attr)
    return wrapped


def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except parameters named ``lora_A`` / ``lora_B``."""
    for name, p in model.named_parameters():
        p.requires_grad_("lora_A" in name or "lora_B" in name)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors: what you ship per task (kilobytes to megabytes)."""
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


class MultiLoRALinear(nn.Module):
    """One frozen base, ``n_adapters`` LoRA pairs, and a per-example adapter id (multi-LoRA serving).

    y_b = x_b W^T + (α/r) x_b A_{id_b}^T B_{id_b}^T — the base matmul is shared across the
    batch; only the small low-rank matmuls are gathered per request (the idea behind
    Punica / S-LoRA style serving).
    """

    def __init__(self, base: nn.Linear, n_adapters: int, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(n_adapters, r, base.in_features) / math.sqrt(base.in_features))  # (n_adapters, r, in)
        self.lora_B = nn.Parameter(torch.zeros(n_adapters, base.out_features, r))  # (n_adapters, out, r)

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in), adapter_ids: (B,) long -> (B, T, out)."""
        A = self.lora_A[adapter_ids]  # (B, r, in)   gather this request's adapter
        Bm = self.lora_B[adapter_ids]  # (B, out, r)
        low = x @ A.transpose(1, 2)  # (B, T, r)
        return self.base(x) + self.scaling * (low @ Bm.transpose(1, 2))  # (B, T, out)
