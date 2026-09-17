# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/perception/box_tokenizer.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k box_tokenizer -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py perception/box_tokenizer --force

"""Detection as sequence generation: boxes ⇄ discrete tokens (NumPy).

Pix2Seq (Chen et al., ICLR 2022), PaliGemma and Florence-2 emit detections as *text*:
each box becomes ``[y_min, x_min, y_max, x_max, class]`` where every coordinate is
quantised to one of ``n_bins`` location tokens (PaliGemma writes them as ``<loc0123>``,
1024 bins).  The vocabulary is ``n_bins`` location tokens, then ``K`` class tokens, then
``EOS``.  Quantisation error per coordinate is at most ``image_size / (2 · n_bins)``.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class BoxTokenizer:
    """Round-trip ``(boxes, labels) → token ids → (boxes, labels)``.

    Boxes are ``(N, 4)`` as ``(x_min, y_min, x_max, y_max)`` in pixels of an
    ``image_hw = (H, W)`` image; labels are ``(N,)`` ints in ``[0, num_classes)``.
    Token layout: ``[0, n_bins)`` = location, ``[n_bins, n_bins + K)`` = class, ``n_bins + K`` = EOS.
    """
    n_bins: int = 1024
    num_classes: int = 80
    image_hw: tuple[int, int] = (1024, 1024)

    @property
    def eos(self) -> int:
        raise NotImplementedError('TODO: implement eos (see the reference in src/mlbook)')

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError('TODO: implement vocab_size (see the reference in src/mlbook)')

    def _quantise(self, value: np.ndarray, extent: float) -> np.ndarray:
        """Continuous coordinate in ``[0, extent]`` → bin in ``[0, n_bins)``."""
        raise NotImplementedError('TODO: implement _quantise (see the reference in src/mlbook)')

    def _dequantise(self, token: np.ndarray, extent: float) -> np.ndarray:
        """Bin → coordinate at the bin centre (inverse of ``_quantise`` up to rounding)."""
        raise NotImplementedError('TODO: implement _dequantise (see the reference in src/mlbook)')

    def encode(self, boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """(N, 4), (N,) → (5·N + 1,) token ids, ordered ``y_min x_min y_max x_max class``, then EOS."""
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Token ids → (boxes (M, 4), labels (M,)).  Stops at EOS; drops a malformed tail.

        A quintuple is kept only if its four coordinates are location tokens and its fifth is
        a class token — the same validity filter a decoder applies to sampled text.
        """
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

    def to_text(self, boxes: np.ndarray, labels: np.ndarray, class_names: list[str]) -> str:
        """PaliGemma-style string: ``<loc0123><loc0045><loc0800><loc0900> car ; ...``."""
        raise NotImplementedError('TODO: implement to_text (see the reference in src/mlbook)')
