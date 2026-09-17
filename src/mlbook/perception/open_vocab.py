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
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_embed)
        self.proj = nn.Linear(d_embed, d_embed)

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.tok(token_ids)  # (K, L, d)
        m = mask.unsqueeze(-1).to(e.dtype)  # (K, L, 1)
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1.0)  # (K, d) mean over real tokens
        return F.normalize(self.proj(pooled), dim=-1)  # (K, d)


class OpenVocabHead(nn.Module):
    """Region features · text embeddings → per-phrase logits.

    forward(region_feats (B, N, d_region), text_emb (K, d_embed)) → (B, N, K).
    ``τ`` is learned in log space (as in CLIP); ``bias`` shifts the sigmoid operating point
    so that untrained phrases start "off" (GLIP/OWL-ViT initialise it to a negative prior).
    """

    def __init__(self, d_region: int, d_embed: int, init_temperature: float = 0.07, bias_init: float = -4.0):
        super().__init__()
        self.proj = nn.Linear(d_region, d_embed)  # region → joint embedding space
        self.log_inv_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / init_temperature)))))
        self.bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, region_feats: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        r = F.normalize(self.proj(region_feats), dim=-1)  # (B, N, d_embed)
        t = F.normalize(text_emb, dim=-1)  # (K, d_embed)
        sim = torch.matmul(r, t.t())  # (B, N, K)  cosine similarity r_i · t_k
        return sim * self.log_inv_temp.exp() + self.bias  # (B, N, K)


def region_word_alignment_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Sigmoid focal loss over every (region, phrase) pair (GLIP / Grounding DINO style).

    Args:
        logits: (B, N, K).  targets: (B, N, K) in {0, 1}; a region may match several phrases
        ("a red car" and "car") or none (background = all zeros).
    Returns:
        scalar, normalised by the number of positive pairs (min 1).
    """
    p = torch.sigmoid(logits)  # (B, N, K)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, N, K)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)  # (B, N, K) prob of the true label
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)  # (B, N, K)
    focal = alpha_t * (1.0 - p_t) ** gamma * ce  # (B, N, K)
    return focal.sum() / targets.sum().clamp(min=1.0)


def zero_shot_classify(logits: torch.Tensor, threshold: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """argmax phrase per region and whether its logit clears ``threshold`` (else background).

    logits (B, N, K) → (labels (B, N) long, is_object (B, N) bool).
    """
    best, labels = logits.max(dim=-1)  # (B, N) each
    return labels, best > threshold
