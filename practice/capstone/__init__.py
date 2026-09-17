"""Canon #60 -- a tiny end-to-end multimodal agent, built from scratch.

The capstone of the [coding canon](../../../docs/part16-coding-canon/index.md):
a vision encoder, a projector, a causal LM, the post-training stages that turn
them into an assistant, and the agent loop that lets it call a tool. Everything
trains on a CPU in seconds on a synthetic grounded task whose answers are
computed from the scene generator, so every reward is verifiable offline.

The package is deliberately self-contained: it imports nothing from the rest of
``mlbook``, so you can read it front to back without jumping between areas, and
so you can retype it in an interview without remembering where anything lives.

Read it in this order:

1. ``synthetic_task``      the world, the vocabulary and the programmatic oracle
2. ``tiny_vision_encoder`` patch embedding plus attention blocks, (B, N_v, d_v)
3. ``projector``           (B, N_v, d_v) -> (B, N_q, d_llm)
4. ``tiny_lm``             an explicit causal Transformer with greedy and sampled decoding
5. ``multimodal_model``    splice visual tokens in front of text, build masks and positions
6. ``sft_stage``           assistant-only cross-entropy
7. ``reward_stage``        Bradley-Terry reward model and the verifier
8. ``preference_stage``    DPO and a simplified GRPO
9. ``tool_loop``           observe, reason, act, observe again
10. ``pipeline``           ``run_pipeline()`` runs all of it and returns a report
"""

from mlbook.capstone import (
    multimodal_model,
    pipeline,
    preference_stage,
    projector,
    reward_stage,
    sft_stage,
    synthetic_task,
    tiny_lm,
    tiny_vision_encoder,
    tool_loop,
)

__all__ = [
    "multimodal_model",
    "pipeline",
    "preference_stage",
    "projector",
    "reward_stage",
    "sft_stage",
    "synthetic_task",
    "tiny_lm",
    "tiny_vision_encoder",
    "tool_loop",
]
