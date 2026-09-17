# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/capstone/multimodal_model.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k multimodal_model -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py capstone/multimodal_model --force

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

    def __init__(self, vocab_size: int, image_size: int=16, patch_size: int=4, in_channels: int=3, d_v: int=32, vision_heads: int=2, vision_layers: int=2, d_llm: int=48, lm_heads: int=4, lm_layers: int=2, n_query_tokens: int=8, max_text_len: int=16) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def visual_embeds(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` -> ``(B, N_q, d)`` tokens the LM can consume."""
        raise NotImplementedError('TODO: implement visual_embeds (see the reference in src/mlbook)')

    def build_inputs(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Splice the two modalities into one sequence.

        Returns ``(inputs_embeds (B, S, d), attention_mask (B, S), position_ids (B, S))``.
        """
        raise NotImplementedError('TODO: implement build_inputs (see the reference in src/mlbook)')

    def hidden_states(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        """``(B, S, d)`` final hidden states over the spliced sequence."""
        raise NotImplementedError('TODO: implement hidden_states (see the reference in src/mlbook)')

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        """``(B, S, V)`` logits over the whole spliced sequence; entry ``i`` predicts position ``i+1``."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def text_logits(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None=None) -> torch.Tensor:
        """Teacher-forcing view: ``(B, T, V)`` where entry ``t`` scores **text token ``t``**.

        The slice ``[N_q - 1 : N_q - 1 + T]`` undoes the next-token shift *and*
        the visual offset in one step, so every loss downstream can index text
        positions directly.
        """
        raise NotImplementedError('TODO: implement text_logits (see the reference in src/mlbook)')

    @torch.no_grad()
    def generate(self, images: torch.Tensor, prompt_ids: torch.Tensor, max_new_tokens: int=2, greedy: bool=True, temperature: float=1.0, top_k: int | None=None, eos_id: int | None=None, attention_mask: torch.Tensor | None=None, generator: torch.Generator | None=None) -> torch.Tensor:
        """Answer a question about an image. ``(B, T_p)`` prompt -> ``(B, max_new_tokens)`` new ids.

        The next token after the prompt is scored by the **last** logit of the
        spliced sequence, which is why this uses :meth:`forward` and not
        :meth:`text_logits`.
        """
        raise NotImplementedError('TODO: implement generate (see the reference in src/mlbook)')

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    raise NotImplementedError('TODO: implement count_parameters (see the reference in src/mlbook)')
