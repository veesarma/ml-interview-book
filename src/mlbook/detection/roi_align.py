"""ROIAlign (He et al. 2017) with explicit bilinear sampling, and ROIPool for contrast (PyTorch).

Given a feature map ``(C, H, W)`` at stride ``s`` and a box in image coordinates, produce a
fixed ``(C, P, P)`` crop.  ROIPool *quantises* the box to the feature grid twice (box edges,
bin edges); ROIAlign keeps continuous coordinates and bilinearly samples, so the crop is
aligned to the box to sub-pixel precision — which is what pixel-accurate masks need.
"""

from __future__ import annotations

import torch


def bilinear_at(feat: torch.Tensor, y: float, x: float) -> torch.Tensor:
    """Bilinear sample of ``feat`` (C, H, W) at continuous ``(y, x)``; zero outside. Returns (C,)."""
    C, H, W = feat.shape
    if y < -1.0 or y > H or x < -1.0 or x > W:
        return feat.new_zeros(C)
    y = min(max(y, 0.0), H - 1.0)
    x = min(max(x, 0.0), W - 1.0)
    y0, x0 = int(y), int(x)
    y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
    ay, ax = y - y0, x - x0  # fractional parts
    top = (1 - ax) * feat[:, y0, x0] + ax * feat[:, y0, x1]  # (C,)
    bottom = (1 - ax) * feat[:, y1, x0] + ax * feat[:, y1, x1]  # (C,)
    return (1 - ay) * top + ay * bottom  # (C,)


def roi_align(feat: torch.Tensor, box: torch.Tensor, output_size: int, spatial_scale: float, sampling_ratio: int = 2) -> torch.Tensor:
    """ROIAlign for a single box.

    Args:
        feat: (C, H, W) feature map.  box: (4,) xyxy in *image* pixels.
        spatial_scale: 1/stride of the feature map.  sampling_ratio: samples per bin side.
    Returns:
        (C, P, P) with P = output_size; each bin is the mean of its ``r×r`` bilinear samples.
    """
    x1, y1, x2, y2 = (box * spatial_scale).tolist()  # continuous feature-map coordinates (no rounding)
    roi_w = max(x2 - x1, 1.0)
    roi_h = max(y2 - y1, 1.0)
    bin_w = roi_w / output_size
    bin_h = roi_h / output_size
    r = sampling_ratio
    out = feat.new_zeros(feat.shape[0], output_size, output_size)  # (C, P, P)
    for i in range(output_size):
        for j in range(output_size):
            acc = feat.new_zeros(feat.shape[0])  # (C,)
            for si in range(r):
                for sj in range(r):
                    y = y1 + i * bin_h + (si + 0.5) * bin_h / r  # sample point, bin interior
                    x = x1 + j * bin_w + (sj + 0.5) * bin_w / r
                    acc = acc + bilinear_at(feat, y, x)
            out[:, i, j] = acc / (r * r)
    return out


def roi_pool(feat: torch.Tensor, box: torch.Tensor, output_size: int, spatial_scale: float) -> torch.Tensor:
    """ROIPool (Fast R-CNN): quantise box to integer cells, max-pool each (integer) bin. (C, P, P)."""
    x1, y1, x2, y2 = [int(round(v)) for v in (box * spatial_scale).tolist()]  # first quantisation
    roi_w = max(x2 - x1 + 1, 1)
    roi_h = max(y2 - y1 + 1, 1)
    out = feat.new_zeros(feat.shape[0], output_size, output_size)  # (C, P, P)
    for i in range(output_size):
        for j in range(output_size):
            r0 = y1 + int(i * roi_h / output_size)  # second quantisation: bin edges
            r1 = y1 + int((i + 1) * roi_h / output_size)
            c0 = x1 + int(j * roi_w / output_size)
            c1 = x1 + int((j + 1) * roi_w / output_size)
            r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)
            r0, c0 = max(r0, 0), max(c0, 0)
            r1, c1 = min(r1, feat.shape[1]), min(c1, feat.shape[2])
            out[:, i, j] = feat[:, r0:r1, c0:c1].amax(dim=(1, 2))  # (C,)
    return out
