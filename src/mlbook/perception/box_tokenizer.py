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
        return self.n_bins + self.num_classes

    @property
    def vocab_size(self) -> int:
        return self.n_bins + self.num_classes + 1

    def _quantise(self, value: np.ndarray, extent: float) -> np.ndarray:
        """Continuous coordinate in ``[0, extent]`` → bin in ``[0, n_bins)``."""
        return np.clip(np.round(value / extent * (self.n_bins - 1)), 0, self.n_bins - 1).astype(np.int64)  # (N,)

    def _dequantise(self, token: np.ndarray, extent: float) -> np.ndarray:
        """Bin → coordinate at the bin centre (inverse of ``_quantise`` up to rounding)."""
        return token.astype(np.float64) / (self.n_bins - 1) * extent  # (N,)

    def encode(self, boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """(N, 4), (N,) → (5·N + 1,) token ids, ordered ``y_min x_min y_max x_max class``, then EOS."""
        h, w = self.image_hw
        y0 = self._quantise(boxes[:, 1], h)  # (N,)
        x0 = self._quantise(boxes[:, 0], w)  # (N,)
        y1 = self._quantise(boxes[:, 3], h)  # (N,)
        x1 = self._quantise(boxes[:, 2], w)  # (N,)
        cls = labels.astype(np.int64) + self.n_bins  # (N,)
        seq = np.stack([y0, x0, y1, x1, cls], axis=1).reshape(-1)  # (5·N,)
        return np.concatenate([seq, np.array([self.eos], dtype=np.int64)])  # (5·N + 1,)

    def decode(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Token ids → (boxes (M, 4), labels (M,)).  Stops at EOS; drops a malformed tail.

        A quintuple is kept only if its four coordinates are location tokens and its fifth is
        a class token — the same validity filter a decoder applies to sampled text.
        """
        h, w = self.image_hw
        toks = np.asarray(tokens, dtype=np.int64)
        eos_pos = np.flatnonzero(toks == self.eos)
        if eos_pos.size:
            toks = toks[: eos_pos[0]]
        n = toks.size // 5
        q = toks[: 5 * n].reshape(n, 5)  # (n, 5)
        is_loc = (q[:, :4] >= 0) & (q[:, :4] < self.n_bins)  # (n, 4)
        is_cls = (q[:, 4] >= self.n_bins) & (q[:, 4] < self.eos)  # (n,)
        keep = is_loc.all(axis=1) & is_cls  # (n,)
        q = q[keep]  # (M, 5)
        boxes = np.stack([
            self._dequantise(q[:, 1], w), self._dequantise(q[:, 0], h),
            self._dequantise(q[:, 3], w), self._dequantise(q[:, 2], h),
        ], axis=1)  # (M, 4) as x_min, y_min, x_max, y_max
        return boxes, q[:, 4] - self.n_bins  # (M, 4), (M,)

    def to_text(self, boxes: np.ndarray, labels: np.ndarray, class_names: list[str]) -> str:
        """PaliGemma-style string: ``<loc0123><loc0045><loc0800><loc0900> car ; ...``."""
        toks = self.encode(boxes, labels)[:-1].reshape(-1, 5)  # (N, 5)
        parts = []
        for row in toks:
            locs = "".join(f"<loc{int(t):04d}>" for t in row[:4])
            parts.append(f"{locs} {class_names[int(row[4]) - self.n_bins]}")
        return " ; ".join(parts)
