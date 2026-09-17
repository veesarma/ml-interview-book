# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/interp/activation_patching.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k activation_patching -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py interp/activation_patching --force

"""Activation patching / causal tracing (Meng et al. 2022; Vig et al. 2020).

Run the model on a *clean* input and a *corrupted* input. Copy one activation
(layer l, position t) from the clean run into the corrupted run and measure how
much of the clean behaviour is restored:

    recovery(l, t) = (metric(patched) - metric(corrupt)) / (metric(clean) - metric(corrupt))

where metric = logit of the clean answer minus logit of the corrupted answer at the
final position. A recovery near 1 means (l, t) carries the causally relevant information.
"""
from __future__ import annotations
import torch
from mlbook.interp.logit_lens import TinyResidualLM

def forward_with_patch(model: TinyResidualLM, tokens: torch.Tensor, layer: int, position: int, replacement: torch.Tensor) -> torch.Tensor:
    """Forward pass where residual h_layer[:, position] is overwritten by ``replacement`` (d,).

    tokens: (1, T) -> logits (1, T, V). ``layer`` indexes h_0 .. h_L (0 = embeddings).
    """
    raise NotImplementedError('TODO: implement forward_with_patch (see the reference in src/mlbook)')

def _patch(h: torch.Tensor, position: int, replacement: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError('TODO: implement _patch (see the reference in src/mlbook)')

def logit_diff(logits: torch.Tensor, answer: int, distractor: int) -> float:
    """logit[answer] - logit[distractor] at the last position; logits (1, T, V)."""
    raise NotImplementedError('TODO: implement logit_diff (see the reference in src/mlbook)')

def patching_map(model: TinyResidualLM, clean: torch.Tensor, corrupt: torch.Tensor, answer: int, distractor: int) -> torch.Tensor:
    """Recovery fraction for every (layer, position): (L + 1, T)."""
    raise NotImplementedError('TODO: implement patching_map (see the reference in src/mlbook)')
