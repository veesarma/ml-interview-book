# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/detection/roi_align.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k roi_align -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py detection/roi_align --force

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
    raise NotImplementedError('TODO: implement bilinear_at (see the reference in src/mlbook)')

def roi_align(feat: torch.Tensor, box: torch.Tensor, output_size: int, spatial_scale: float, sampling_ratio: int=2) -> torch.Tensor:
    """ROIAlign for a single box.

    Args:
        feat: (C, H, W) feature map.  box: (4,) xyxy in *image* pixels.
        spatial_scale: 1/stride of the feature map.  sampling_ratio: samples per bin side.
    Returns:
        (C, P, P) with P = output_size; each bin is the mean of its ``r×r`` bilinear samples.
    """
    raise NotImplementedError('TODO: implement roi_align (see the reference in src/mlbook)')

def roi_pool(feat: torch.Tensor, box: torch.Tensor, output_size: int, spatial_scale: float) -> torch.Tensor:
    """ROIPool (Fast R-CNN): quantise box to integer cells, max-pool each (integer) bin. (C, P, P)."""
    raise NotImplementedError('TODO: implement roi_pool (see the reference in src/mlbook)')
