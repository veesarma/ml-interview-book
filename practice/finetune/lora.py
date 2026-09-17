# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/finetune/lora.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k lora -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py finetune/lora --force

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

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float=0.0) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def delta_weight(self) -> torch.Tensor:
        """ΔW = (α/r) · B A, shape (out, in)."""
        raise NotImplementedError('TODO: implement delta_weight (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, out)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight for zero-overhead inference."""
        raise NotImplementedError('TODO: implement merge (see the reference in src/mlbook)')

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo ``merge`` so training can continue on the separate A, B."""
        raise NotImplementedError('TODO: implement unmerge (see the reference in src/mlbook)')

def apply_lora(model: nn.Module, r: int, alpha: float, target_names: tuple[str, ...]=('q_proj', 'v_proj')) -> list[str]:
    """Replace every ``nn.Linear`` whose attribute name is in ``target_names`` with a LoRALinear.

    Returns the dotted names that were wrapped.  Then call ``mark_only_lora_trainable``.
    """
    raise NotImplementedError('TODO: implement apply_lora (see the reference in src/mlbook)')

def mark_only_lora_trainable(model: nn.Module) -> None:
    """Freeze everything except parameters named ``lora_A`` / ``lora_B``."""
    raise NotImplementedError('TODO: implement mark_only_lora_trainable (see the reference in src/mlbook)')

def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors: what you ship per task (kilobytes to megabytes)."""
    raise NotImplementedError('TODO: implement lora_state_dict (see the reference in src/mlbook)')

def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) parameter counts."""
    raise NotImplementedError('TODO: implement count_parameters (see the reference in src/mlbook)')

class MultiLoRALinear(nn.Module):
    """One frozen base, ``n_adapters`` LoRA pairs, and a per-example adapter id (multi-LoRA serving).

    y_b = x_b W^T + (α/r) x_b A_{id_b}^T B_{id_b}^T — the base matmul is shared across the
    batch; only the small low-rank matmuls are gathered per request (the idea behind
    Punica / S-LoRA style serving).
    """

    def __init__(self, base: nn.Linear, n_adapters: int, r: int, alpha: float) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor, adapter_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, T, in), adapter_ids: (B,) long -> (B, T, out)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
