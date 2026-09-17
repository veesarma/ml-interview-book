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
    T = tokens.shape[1]
    h = model.embed(tokens) + model.pos(torch.arange(T))  # (1, T, d)
    if layer == 0:
        h = _patch(h, position, replacement)
    for l, block in enumerate(model.blocks, start=1):
        h = block(h)  # (1, T, d)
        if l == layer:
            h = _patch(h, position, replacement)
    return model.unembed(model.ln_f(h))  # (1, T, V)


def _patch(h: torch.Tensor, position: int, replacement: torch.Tensor) -> torch.Tensor:
    h = h.clone()  # (1, T, d)
    h[0, position] = replacement  # (d,)
    return h


def logit_diff(logits: torch.Tensor, answer: int, distractor: int) -> float:
    """logit[answer] - logit[distractor] at the last position; logits (1, T, V)."""
    last = logits[0, -1]  # (V,)
    return float(last[answer] - last[distractor])


def patching_map(model: TinyResidualLM, clean: torch.Tensor, corrupt: torch.Tensor, answer: int, distractor: int) -> torch.Tensor:
    """Recovery fraction for every (layer, position): (L + 1, T)."""
    with torch.no_grad():
        clean_res = model.residuals(clean)  # list of (1, T, d)
        m_clean = logit_diff(model(clean), answer, distractor)
        m_corr = logit_diff(model(corrupt), answer, distractor)
        L, T = len(clean_res) - 1, clean.shape[1]
        out = torch.zeros(L + 1, T)  # (L+1, T)
        for l in range(L + 1):
            for t in range(T):
                patched = forward_with_patch(model, corrupt, l, t, clean_res[l][0, t])
                out[l, t] = (logit_diff(patched, answer, distractor) - m_corr) / (m_clean - m_corr + 1e-9)
    return out
