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

    def __init__(self, d_v: int, d_llm: int, n_visual_tokens: int, n_query_tokens: int, hidden: int | None = None) -> None:
        super().__init__()
        if n_visual_tokens % n_query_tokens != 0:
            raise ValueError(f"{n_visual_tokens} visual tokens do not group evenly into {n_query_tokens}")
        self.group = n_visual_tokens // n_query_tokens
        self.n_query_tokens = n_query_tokens
        in_dim = self.group * d_v
        hidden = 2 * d_llm if hidden is None else hidden
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_llm)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        B, N_v, d_v = visual_tokens.shape                                # (B, N_v, d_v)
        x = visual_tokens.reshape(B, self.n_query_tokens, self.group * d_v)  # (B, N_q, group*d_v)
        x = self.act(self.fc1(x))                                        # (B, N_q, hidden)
        return self.fc2(x)                                               # (B, N_q, d_llm)
