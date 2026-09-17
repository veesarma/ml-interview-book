# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/open_vocab.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k open_vocab -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/open_vocab --force

"""Open-vocabulary classification by region–text similarity (PyTorch).

Closed-set detectors end in ``nn.Linear(d, K)``: the class "weights" are ``K`` learned rows.
Open-vocabulary detectors (ViLD, GLIP, OWL-ViT, Grounding DINO, YOLO-World) replace those
rows with **text embeddings**: for region feature ``r_i ∈ R^d`` and phrase embedding
``t_k ∈ R^d`` (both L2-normalised after a projection),

    logit_{ik} = (r_i · t_k) / τ.

Adding a class means adding a row of text — no retraining.  Training uses a per-(region,
phrase) sigmoid loss (GLIP's "word–region alignment"), not a softmax over a fixed K, so
the set of phrases can differ from image to image.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class TinyTextEncoder(nn.Module):
    """Mean-pooled token embeddings + linear projection: stands in for CLIP/BERT text towers.

    forward(token_ids (K, L), mask (K, L)) → (K, d_embed), L2-normalised.
    """

    def __init__(self, vocab_size: int, d_embed: int):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

class OpenVocabHead(nn.Module):
    """Region features · text embeddings → per-phrase logits.

    forward(region_feats (B, N, d_region), text_emb (K, d_embed)) → (B, N, K).
    ``τ`` is learned in log space (as in CLIP); ``bias`` shifts the sigmoid operating point
    so that untrained phrases start "off" (GLIP/OWL-ViT initialise it to a negative prior).
    """

    def __init__(self, d_region: int, d_embed: int, init_temperature: float=0.07, bias_init: float=-4.0):
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def forward(self, region_feats: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

def region_word_alignment_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float=0.25, gamma: float=2.0) -> torch.Tensor:
    """Sigmoid focal loss over every (region, phrase) pair (GLIP / Grounding DINO style).

    Args:
        logits: (B, N, K).  targets: (B, N, K) in {0, 1}; a region may match several phrases
        ("a red car" and "car") or none (background = all zeros).
    Returns:
        scalar, normalised by the number of positive pairs (min 1).
    """
    raise NotImplementedError('TODO: implement region_word_alignment_loss (see the reference in src/mlbook)')

def zero_shot_classify(logits: torch.Tensor, threshold: float=0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """argmax phrase per region and whether its logit clears ``threshold`` (else background).

    logits (B, N, K) → (labels (B, N) long, is_object (B, N) bool).
    """
    raise NotImplementedError('TODO: implement zero_shot_classify (see the reference in src/mlbook)')
