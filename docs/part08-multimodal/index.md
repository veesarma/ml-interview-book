# Part VIII: Vision Transformers and multimodal models

This part is the bridge from perception to vision-language models. If you have shipped
detectors, OCR systems or segmentation models, everything you know about images still
holds. What changes is that the image becomes a sequence of tokens, and a Transformer
built for text can read it. Once that is true, an image, a caption, a video clip, a
spectrogram and a robot action can all be tokens in one sequence, and the next-token
machinery from [Part VI](../part06-llm-training/index.md) produces captions, answers,
boxes and actions.

The six chapters derive that bridge piece by piece.

| Chapter | The idea you must own | The code you must be able to write |
|---|---|---|
| [1. Vision Transformers](01-vision-transformers.md) | An image is $N = HW/P^2$ tokens. Patch embedding is a conv with kernel = stride = $P$. Windowed attention with the cyclic-shift mask makes it hierarchical. | `PatchEmbed`, a ViT with explicit MHA and CLS/GAP heads, `shifted_window_mask`. |
| [2. DETR and set prediction](02-detr.md) | Detection as set prediction: $Q$ object queries, a Hungarian match, a loss with GIoU and a down-weighted "no object" class, no NMS. | Kuhn-Munkres from scratch in NumPy, the DETR matcher and loss. |
| [3. CLIP and contrastive learning](03-clip-contrastive.md) | Symmetric InfoNCE over a $B\times B$ similarity matrix, the temperature, why batch size is the hyperparameter, SigLIP's sigmoid alternative, zero-shot classification. | Mini-CLIP with both losses and zero-shot classification. |
| [4. VLM architecture](04-vlm-architecture.md) | Image, vision encoder, projector, LLM. Four projector designs. How many visual tokens an image costs in prefill and KV cache. | Linear, MLP, resampler and cross-attention projectors, a mini VLM with visual-token insertion. |
| [5. Multimodal foundation models](05-multimodal-foundation.md) | Three strategies (shared embedding, encoders plus shared LLM, everything tokenised) and the decision rule between them. VLAs as the bridge to robotics. | Reading chapter, built on the code from chapters 3 and 4. |
| [6. Video models](06-video-models.md) | Tubelets. Factorised space-time attention and why it costs $O(T(hw)^2 + hw\,T^2)$ instead of $O((Thw)^2)$. Long-video compression. | Tubelet embedding, a factorised space-time block. |

## Prerequisites

* [Attention mathematics](../part05-sequence-transformers/03-attention-mathematics.md) and
  [Transformer architectures](../part05-sequence-transformers/04-transformer-architectures.md).
  This part re-implements a small attention block so its modules stand alone, but it does
  not re-derive attention.
* [Convolutions](../part04-vision/02-convolutions.md) and
  [CNN architectures](../part04-vision/03-cnn-architectures.md). The inductive-bias
  argument in chapter 1 is a comparison against what you already know.
* [Object detection](../part04-vision/04-detection.md). DETR is defined by what it removes
  from the detectors there: anchors and NMS.
* [Efficient attention and the KV cache](../part06-llm-training/04-efficient-attention-kv-cache.md).
  Chapter 4's cost model for visual tokens is that chapter's arithmetic applied to images.
* [Information theory](../part01-math/05-information-theory.md) for InfoNCE's
  mutual-information bound in chapter 3.

## If you have one day

Read in this order and stop when time runs out. Each step stands alone.

1. Chapter 1, sections 1 to 3 (patchification, the ViT, the shifted-window mask), 2 hours.
   Retype `PatchEmbedLinear` and `TinyViT` from memory.
2. Chapter 3, section 2 (symmetric InfoNCE, temperature, SigLIP) and section 3 (mini-CLIP),
   1.5 hours. Retype `clip_loss` and `siglip_loss`.
3. Chapter 4, sections 1 to 4 (the canonical VLM, the four projectors, the token-count cost
   model), 2 hours. Retype `merge_visual_tokens` and one projector.
4. Chapter 2, sections 2 and 3 (matching cost, Hungarian, DETR loss), 1.5 hours. Retype
   `hungarian`.
5. Chapter 5, section 4 (the decision table across the three strategies) and chapter 6,
   section 2 (factorised attention cost), 1 hour. Read only.

## Where this part is used later

* [Part XI, Perception and autonomy](../part11-perception-autonomy/index.md): BEV
  Transformers, occupancy networks and world models are ViTs, DETR-style queries and video
  Transformers applied to multi-camera rigs.
* [Part IX, Generative](../part09-generative/index.md): the latent diffusion Transformers
  mentioned in chapter 6 are derived there.
* [Part XVII, ML system design](../part17-ml-system-design/index.md): the visual search and
  OCR/document-understanding designs are CLIP and VLM systems with a serving budget.

## Code map

All Part VIII code lives in `src/mlbook/multimodal/` and is tested by
`tests/test_multimodal_*.py` (`python -m pytest tests/test_multimodal_* -q`):

| Module | Contents |
|---|---|
| `attention_block.py` | explicit multi-head self- and cross-attention, pre-norm block |
| `patch_embed.py` | `patchify`, linear and conv patch embedding (tested equal), 2-D sinusoidal and interpolated position embeddings |
| `vit.py` | `TinyViT` (CLS or GAP, registers), `DistillableViT` (DeiT token) |
| `swin_window.py` | window partition and reverse, `shifted_window_mask`, relative position bias, `SwinBlock`, `PatchMerging` |
| `hungarian.py` | Kuhn-Munkres in NumPy, tested against `scipy.optimize.linear_sum_assignment` |
| `detr_loss.py` | pairwise GIoU, matching cost, `hungarian_match`, `detr_loss` |
| `clip.py` | `MiniCLIP`, `clip_loss`, `siglip_loss`, `zero_shot_classify` |
| `projectors.py` | `LinearProjector`, `MLPProjector`, `PerceiverResampler`, `QFormer`, `GatedCrossAttentionAdapter` |
| `vlm.py` | `ToyVisionEncoder`, `TinyCausalLM`, `merge_visual_tokens`, `MiniVLM`, `vlm_lm_loss` |
| `token_compression.py` | pooling, pixel-shuffle merge, score-based pruning, AnyRes tiling, `visual_token_count` |
| `video_attention.py` | tubelet embedding, `FactorisedSpaceTimeBlock`, `JointSpaceTimeBlock`, `attention_flops` |
