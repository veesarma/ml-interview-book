"""Canon #60, part 4 -- the VLM: splice visual tokens in front of the text tokens.

This is the whole of the LLaVA-style architecture in one class. The only
subtlety worth an interview is *bookkeeping*, and it is done explicitly here:

* the visual tokens occupy absolute positions ``0 .. N_q-1`` and the text tokens
  ``N_q .. N_q+T-1``, so ``position_ids`` is built by hand;
* the attention mask is the text mask with ``N_q`` ones prepended, because the
  image is never padding;
* ``logits[:, i]`` predicts the token at ``i + 1``, so the logit that scores
  text token ``t`` lives at index ``N_q + t - 1``. Getting that off by one is the
  single most common bug in a from-scratch VLM.

Shapes: ``B`` batch, ``N_q`` visual tokens after the projector, ``T`` text
tokens, ``S = N_q + T`` total, ``d`` LLM width, ``V`` vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlbook.capstone.projector import Projector
from mlbook.capstone.tiny_lm import TinyCausalLM, sample_next_token
from mlbook.capstone.tiny_vision_encoder import TinyVisionEncoder


class TinyVLM(nn.Module):
    """Vision encoder + projector + causal LM, trained jointly (nothing is frozen here)."""

    def __init__(
        self,
        vocab_size: int,
        image_size: int = 16,
        patch_size: int = 4,
        in_channels: int = 3,
        d_v: int = 32,
        vision_heads: int = 2,
        vision_layers: int = 2,
        d_llm: int = 48,
        lm_heads: int = 4,
        lm_layers: int = 2,
        n_query_tokens: int = 8,
        max_text_len: int = 16,
    ) -> None:
        super().__init__()
        self.vision = TinyVisionEncoder(image_size, patch_size, in_channels, d_v, vision_heads, vision_layers)
        self.projector = Projector(d_v, d_llm, self.vision.n_tokens, n_query_tokens)
        self.lm = TinyCausalLM(
            vocab_size=vocab_size,
            d_model=d_llm,
            n_heads=lm_heads,
            n_layers=lm_layers,
            max_positions=n_query_tokens + max_text_len,
        )
        self.n_query_tokens = n_query_tokens
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size

    # -- assembling the input sequence ----------------------------------------

    def visual_embeds(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, N_q, d)`` tokens the LM can consume."""
        visual_tokens = self.vision(images)                              # (B, N_v, d_v)
        return self.projector(visual_tokens)                             # (B, N_q, d)

    def build_inputs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splice the two modalities into one sequence.

        Returns ``(inputs_embeds (B, S, d), attention_mask (B, S), position_ids (B, S))``.
        """
        B, T = input_ids.shape                                           # (B, T)
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)  # (B, T)
        vis = self.visual_embeds(images)                                 # (B, N_q, d)
        txt = self.lm.embed_tokens(input_ids)                            # (B, T, d)
        inputs_embeds = torch.cat([vis, txt], dim=1)                     # (B, S, d)
        vis_mask = torch.ones(B, self.n_query_tokens, dtype=attention_mask.dtype, device=device)  # (B, N_q)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)         # (B, S)
        S = self.n_query_tokens + T
        position_ids = torch.arange(S, device=device).expand(B, S)       # (B, S)
        return inputs_embeds, full_mask, position_ids

    # -- forward passes --------------------------------------------------------

    def hidden_states(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, S, d)`` final hidden states over the spliced sequence."""
        inputs_embeds, full_mask, position_ids = self.build_inputs(images, input_ids, attention_mask)
        return self.lm.encode(inputs_embeds, full_mask, position_ids)    # (B, S, d)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, S, V)`` logits over the whole spliced sequence; entry ``i`` predicts position ``i+1``."""
        h = self.hidden_states(images, input_ids, attention_mask)        # (B, S, d)
        return self.lm.lm_head(h)                                        # (B, S, V)

    def text_logits(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forcing view: ``(B, T, V)`` where entry ``t`` scores **text token ``t``**.

        The slice ``[N_q - 1 : N_q - 1 + T]`` undoes the next-token shift *and*
        the visual offset in one step, so every loss downstream can index text
        positions directly.
        """
        logits = self.forward(images, input_ids, attention_mask)         # (B, S, V)
        T = input_ids.shape[1]
        start = self.n_query_tokens - 1
        return logits[:, start:start + T, :]                             # (B, T, V)

    # -- generation ------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 2,
        greedy: bool = True,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Answer a question about an image. ``(B, T_p)`` prompt -> ``(B, max_new_tokens)`` new ids.

        The next token after the prompt is scored by the **last** logit of the
        spliced sequence, which is why this uses :meth:`forward` and not
        :meth:`text_logits`.
        """
        ids = prompt_ids                                                 # (B, T_p)
        mask = torch.ones_like(ids) if attention_mask is None else attention_mask  # (B, T_p)
        generated = []
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)  # (B,)
        for _ in range(max_new_tokens):
            logits = self.forward(images, ids, mask)[:, -1, :]           # (B, V)
            next_id = sample_next_token(logits, greedy, temperature, top_k, generator)  # (B,)
            if eos_id is not None:
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)  # (B,)
                finished = finished | (next_id == eos_id)                # (B,)
            generated.append(next_id)
            ids = torch.cat([ids, next_id[:, None]], dim=1)              # (B, T_p + 1)
            mask = torch.cat([mask, torch.ones_like(next_id)[:, None]], dim=1)  # (B, T_p + 1)
        return torch.stack(generated, dim=1)                             # (B, max_new_tokens)


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
