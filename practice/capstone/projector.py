# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/projector.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k projector -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/projector --force

"""Canon #60, part 2 -- the projector: the two-line module that makes a VLM a VLM.

The projector is the only new parameter block in the LLaVA recipe. It does two
jobs at once and both are interview questions:

1. **Change the width.** ``d_v`` (vision) and ``d_llm`` (language) are different
   numbers chosen by different teams; an MLP maps one to the other.
2. **Choose the token budget.** ``N_v`` patches become ``N_q`` language tokens.
   Every visual token you keep costs LLM context and attention FLOPs forever
   after, so the compression factor ``N_v / N_q`` is a real product decision,
   not a detail. Here it is done by *concatenating* groups of adjacent patches
   before the MLP (the "pixel-shuffle" trick used by InternVL), which keeps all
   the information and pays for it in width rather than throwing it away like
   average pooling would.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class Projector(nn.Module):
    """MLP projector with patch-group concatenation.

    Input  ``(B, N_v, d_v)``.
    Output ``(B, N_q, d_llm)`` where ``N_q = N_v // group``.

    Implements ``h_j = GELU([z_{jg}, ..., z_{jg+g-1}] W_1 + b_1) W_2 + b_2``.
    """

    def __init__(self, d_v: int, d_llm: int, n_visual_tokens: int, n_query_tokens: int, hidden: int | None=None) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')
