# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/image_ops.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k image_ops -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/image_ops --force

"""Image representation and classical filtering, in pure NumPy.

Conventions
-----------
* A grayscale image is ``(H, W)``; a multi-channel image is ``(C, H, W)``
  (channels-first, matching PyTorch).  Pixel ``(i, j)`` means row ``i`` (y),
  column ``j`` (x).  Continuous coordinates put the *centre* of pixel ``(i, j)``
  at ``(y=i, x=j)`` unless a function says otherwise.
* "Convolution" in deep-learning libraries is mathematically *cross-correlation*;
  :func:`correlate2d` is what ``torch.nn.functional.conv2d`` computes and
  :func:`convolve2d` is the textbook operation (kernel flipped).  The two agree
  for symmetric kernels (Gaussian, Laplacian) and differ in sign for
  antisymmetric ones (Sobel).
"""
from __future__ import annotations
import numpy as np

def gaussian_kernel_1d(sigma: float, radius: int | None=None) -> np.ndarray:
    """Sampled, normalised 1-D Gaussian ``g[t] ∝ exp(-t² / 2σ²)``.

    Args:
        sigma: standard deviation in pixels.
        radius: half-width; defaults to ``ceil(3σ)`` (captures 99.7% of mass).
    Returns:
        (2r+1,) kernel that sums to one.
    """
    raise NotImplementedError('TODO: implement gaussian_kernel_1d (see the reference in src/mlbook)')

def gaussian_kernel_2d(sigma: float, radius: int | None=None) -> np.ndarray:
    """Separable 2-D Gaussian ``G[i, j] = g[i] g[j]`` (outer product).

    Returns:
        (2r+1, 2r+1) kernel that sums to one.
    """
    raise NotImplementedError('TODO: implement gaussian_kernel_2d (see the reference in src/mlbook)')
SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.T.copy()
LAPLACIAN_4 = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

def pad2d(img: np.ndarray, pad_h: int, pad_w: int, mode: str='reflect') -> np.ndarray:
    """Pad the last two axes of ``img`` by ``pad_h`` rows and ``pad_w`` columns.

    ``mode`` is any ``np.pad`` mode ("reflect", "constant", "wrap", "edge").
    Returns:
        (..., H + 2 pad_h, W + 2 pad_w)
    """
    raise NotImplementedError('TODO: implement pad2d (see the reference in src/mlbook)')

def correlate2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str='reflect') -> np.ndarray:
    """Cross-correlation ``out[i, j] = Σ_{u,v} img[i+u-r_h, j+v-r_w] · kernel[u, v]``.

    This is exactly what ``F.conv2d`` computes (no kernel flip).  The loop is over
    the ``k_h · k_w`` kernel taps, each step vectorised over all pixels, so the
    cost is ``O(H W k_h k_w)`` with only ``k_h k_w`` Python iterations.

    Args:
        img: (H, W) or (C, H, W).
        kernel: (k_h, k_w) with odd sides.
    Returns:
        same shape as ``img`` ("same" output).
    """
    raise NotImplementedError('TODO: implement correlate2d (see the reference in src/mlbook)')

def convolve2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str='reflect') -> np.ndarray:
    """True convolution ``out = img ∗ kernel`` = correlation with the flipped kernel."""
    raise NotImplementedError('TODO: implement convolve2d (see the reference in src/mlbook)')

def separable_filter(img: np.ndarray, k_col: np.ndarray, k_row: np.ndarray, pad_mode: str='reflect') -> np.ndarray:
    """Filter with a rank-1 kernel ``K = k_col ⊗ k_row`` in two 1-D passes.

    Cost falls from ``O(H W k²)`` to ``O(H W · 2k)``.

    Args:
        img: (H, W) or (C, H, W).
        k_col: (k_h,) applied down the rows (axis -2).
        k_row: (k_w,) applied along the columns (axis -1).
    """
    raise NotImplementedError('TODO: implement separable_filter (see the reference in src/mlbook)')

def gaussian_blur(img: np.ndarray, sigma: float, pad_mode: str='reflect') -> np.ndarray:
    """Separable Gaussian blur of a (H, W) or (C, H, W) image."""
    raise NotImplementedError('TODO: implement gaussian_blur (see the reference in src/mlbook)')

def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel gradients by *correlation* (as OpenCV's ``filter2D``/``Sobel`` do), so ``gx > 0``
    where intensity rises to the right.  Under true convolution the sign would flip —
    the classic gotcha with antisymmetric kernels.

    Returns:
        (gx, gy, magnitude), each the same shape as ``img``.
    """
    raise NotImplementedError('TODO: implement sobel (see the reference in src/mlbook)')

def laplacian(img: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian ``∇²I ≈ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)-4I(i,j)``."""
    raise NotImplementedError('TODO: implement laplacian (see the reference in src/mlbook)')

def fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution via the convolution theorem: ``F(img ∗ k) = F(img) · F(k)``.

    Matches :func:`convolve2d` with ``pad_mode="wrap"`` for odd kernels.

    Args:
        img: (H, W) real image.
        kernel: (k_h, k_w) odd-sized kernel, centred.
    """
    raise NotImplementedError('TODO: implement fft_convolve2d (see the reference in src/mlbook)')

def bilinear_sample(img: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample ``img`` at continuous pixel coordinates ``(ys, xs)`` (pixel centres at integers).

    ``I(y, x) = (1-a)(1-b) I[y0,x0] + (1-a) b I[y0,x0+1] + a (1-b) I[y0+1,x0] + a b I[y0+1,x0+1]``
    with ``a = y - y0``, ``b = x - x0``.  Coordinates outside the image are clamped
    (equivalent to ``padding_mode="border"`` in ``F.grid_sample``).

    Args:
        img: (H, W) or (C, H, W).
        ys, xs: (N,) float coordinates.
    Returns:
        (N,) or (C, N) sampled values.
    """
    raise NotImplementedError('TODO: implement bilinear_sample (see the reference in src/mlbook)')

def resize_bilinear(img: np.ndarray, out_h: int, out_w: int, align_corners: bool=False) -> np.ndarray:
    """Resize with bilinear interpolation (matches ``F.interpolate(mode="bilinear")``).

    With ``align_corners=False`` (the PyTorch/OpenCV default) source coordinate is
    ``src = (dst + 0.5) · (H_in / H_out) − 0.5``: pixel *centres* are aligned, so the
    image content is not shifted by half a pixel.

    Args:
        img: (H, W) or (C, H, W).
    Returns:
        (out_h, out_w) or (C, out_h, out_w).
    """
    raise NotImplementedError('TODO: implement resize_bilinear (see the reference in src/mlbook)')

def downsample2(img: np.ndarray, sigma: float=1.0) -> np.ndarray:
    """Anti-aliased ×2 downsampling: Gaussian blur (the low-pass) then keep even pixels.

    Blurring first removes frequencies above the new Nyquist limit, so decimation
    cannot fold them back into low frequencies (aliasing).
    """
    raise NotImplementedError('TODO: implement downsample2 (see the reference in src/mlbook)')

def gaussian_pyramid(img: np.ndarray, levels: int, sigma: float=1.0) -> list[np.ndarray]:
    """``[G_0 = img, G_1 = down(G_0), …, G_{levels-1}]`` each half the previous size."""
    raise NotImplementedError('TODO: implement gaussian_pyramid (see the reference in src/mlbook)')

def laplacian_pyramid(img: np.ndarray, levels: int, sigma: float=1.0) -> list[np.ndarray]:
    """Band-pass pyramid ``L_l = G_l − up(G_{l+1})``; the last level is the residual ``G_{levels-1}``.

    Because ``G_l = L_l + up(G_{l+1})``, the image is reconstructed *exactly* by
    :func:`reconstruct_from_laplacian` — it is a lossless, multi-scale representation.
    """
    raise NotImplementedError('TODO: implement laplacian_pyramid (see the reference in src/mlbook)')

def reconstruct_from_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    """Invert :func:`laplacian_pyramid` exactly: ``G_l = L_l + up(G_{l+1})`` from the top."""
    raise NotImplementedError('TODO: implement reconstruct_from_laplacian (see the reference in src/mlbook)')

def blur_pool(x: np.ndarray, stride: int=2, filt_size: int=3) -> np.ndarray:
    """BlurPool (Zhang, 2019): binomial low-pass filter, then subsample.

    Replaces a plain stride-``s`` subsample with ``blur → stride-s`` so that small input
    shifts do not flip the output (approximate shift-equivariance).

    Args:
        x: (C, H, W) or (H, W).
        filt_size: 2 → [1,1], 3 → [1,2,1], 5 → [1,4,6,4,1] (rows of Pascal's triangle).
    Returns:
        (..., ceil(H/s), ceil(W/s))
    """
    raise NotImplementedError('TODO: implement blur_pool (see the reference in src/mlbook)')

def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma ``Y = 0.299 R + 0.587 G + 0.114 B``.  ``rgb``: (3, H, W) → (H, W)."""
    raise NotImplementedError('TODO: implement rgb_to_gray (see the reference in src/mlbook)')

def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """BT.601 analog YUV: luma plus two chroma differences (B−Y, R−Y) scaled.

    ``rgb``: (3, H, W) in [0, 1] → (3, H, W) with Y ∈ [0,1], U ∈ ±0.436, V ∈ ±0.615.
    """
    raise NotImplementedError('TODO: implement rgb_to_yuv (see the reference in src/mlbook)')

def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_yuv`. ``yuv``: (3, H, W) → (3, H, W)."""
    raise NotImplementedError('TODO: implement yuv_to_rgb (see the reference in src/mlbook)')

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Hue ∈ [0, 1), saturation ∈ [0, 1], value ∈ [0, 1] from ``rgb`` (3, H, W) in [0, 1]."""
    raise NotImplementedError('TODO: implement rgb_to_hsv (see the reference in src/mlbook)')

def bayer_mosaic(rgb: np.ndarray) -> np.ndarray:
    """Simulate an RGGB Bayer sensor: one colour sample per pixel.

    ``rgb``: (3, H, W) → (H, W) raw mosaic.  Even rows alternate R,G; odd rows G,B.
    Demosaicing (interpolating the two missing channels) is the ISP's first job.
    """
    raise NotImplementedError('TODO: implement bayer_mosaic (see the reference in src/mlbook)')
