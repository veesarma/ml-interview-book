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

# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Sampled, normalised 1-D Gaussian ``g[t] ∝ exp(-t² / 2σ²)``.

    Args:
        sigma: standard deviation in pixels.
        radius: half-width; defaults to ``ceil(3σ)`` (captures 99.7% of mass).
    Returns:
        (2r+1,) kernel that sums to one.
    """
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))
    t = np.arange(-radius, radius + 1, dtype=np.float64)  # (2r+1,)
    g = np.exp(-0.5 * (t / sigma) ** 2)  # (2r+1,)
    return g / g.sum()  # (2r+1,)


def gaussian_kernel_2d(sigma: float, radius: int | None = None) -> np.ndarray:
    """Separable 2-D Gaussian ``G[i, j] = g[i] g[j]`` (outer product).

    Returns:
        (2r+1, 2r+1) kernel that sums to one.
    """
    g = gaussian_kernel_1d(sigma, radius)  # (k,)
    return np.outer(g, g)  # (k, k)


SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])  # (3, 3)
SOBEL_Y = SOBEL_X.T.copy()  # (3, 3)
LAPLACIAN_4 = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])  # (3, 3)

# ---------------------------------------------------------------------------
# Padding and 2-D filtering
# ---------------------------------------------------------------------------


def pad2d(img: np.ndarray, pad_h: int, pad_w: int, mode: str = "reflect") -> np.ndarray:
    """Pad the last two axes of ``img`` by ``pad_h`` rows and ``pad_w`` columns.

    ``mode`` is any ``np.pad`` mode ("reflect", "constant", "wrap", "edge").
    Returns:
        (..., H + 2 pad_h, W + 2 pad_w)
    """
    widths = [(0, 0)] * (img.ndim - 2) + [(pad_h, pad_h), (pad_w, pad_w)]
    return np.pad(img, widths, mode=mode)  # (..., H+2ph, W+2pw)


def correlate2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
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
    k_h, k_w = kernel.shape
    r_h, r_w = k_h // 2, k_w // 2
    padded = pad2d(img, r_h, r_w, pad_mode)  # (..., H+2r_h, W+2r_w)
    H, W = img.shape[-2:]
    out = np.zeros_like(img, dtype=np.float64)  # (..., H, W)
    for u in range(k_h):
        for v in range(k_w):
            # shifted view of the padded image aligned with output pixel (i, j)
            window = padded[..., u : u + H, v : v + W]  # (..., H, W)
            out += kernel[u, v] * window
    return out


def convolve2d(img: np.ndarray, kernel: np.ndarray, pad_mode: str = "reflect") -> np.ndarray:
    """True convolution ``out = img ∗ kernel`` = correlation with the flipped kernel."""
    flipped = kernel[::-1, ::-1]  # (k_h, k_w) flipped in both axes
    return correlate2d(img, flipped, pad_mode)


def separable_filter(
    img: np.ndarray, k_col: np.ndarray, k_row: np.ndarray, pad_mode: str = "reflect"
) -> np.ndarray:
    """Filter with a rank-1 kernel ``K = k_col ⊗ k_row`` in two 1-D passes.

    Cost falls from ``O(H W k²)`` to ``O(H W · 2k)``.

    Args:
        img: (H, W) or (C, H, W).
        k_col: (k_h,) applied down the rows (axis -2).
        k_row: (k_w,) applied along the columns (axis -1).
    """
    tmp = correlate2d(img, k_row[None, :], pad_mode)  # (..., H, W)  horizontal pass
    return correlate2d(tmp, k_col[:, None], pad_mode)  # (..., H, W)  vertical pass


def gaussian_blur(img: np.ndarray, sigma: float, pad_mode: str = "reflect") -> np.ndarray:
    """Separable Gaussian blur of a (H, W) or (C, H, W) image."""
    g = gaussian_kernel_1d(sigma)  # (k,)
    return separable_filter(img, g, g, pad_mode)  # (..., H, W)


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobel gradients by *convolution* (so ``gx > 0`` where intensity rises to the right).

    Returns:
        (gx, gy, magnitude), each the same shape as ``img``.
    """
    gx = convolve2d(img, SOBEL_X)  # (..., H, W)  ∂I/∂x  (times 8)
    gy = convolve2d(img, SOBEL_Y)  # (..., H, W)  ∂I/∂y
    mag = np.sqrt(gx**2 + gy**2)  # (..., H, W)
    return gx, gy, mag


def laplacian(img: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian ``∇²I ≈ I(i+1,j)+I(i-1,j)+I(i,j+1)+I(i,j-1)-4I(i,j)``."""
    return convolve2d(img, LAPLACIAN_4)  # (..., H, W)


# ---------------------------------------------------------------------------
# Fourier view
# ---------------------------------------------------------------------------


def fft_convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular convolution via the convolution theorem: ``F(img ∗ k) = F(img) · F(k)``.

    Matches :func:`convolve2d` with ``pad_mode="wrap"`` for odd kernels.

    Args:
        img: (H, W) real image.
        kernel: (k_h, k_w) odd-sized kernel, centred.
    """
    H, W = img.shape
    k_h, k_w = kernel.shape
    padded_k = np.zeros((H, W), dtype=np.float64)  # (H, W)
    padded_k[:k_h, :k_w] = kernel
    # roll so the kernel centre sits at index (0, 0): circular convolution is centred
    padded_k = np.roll(padded_k, shift=(-(k_h // 2), -(k_w // 2)), axis=(0, 1))  # (H, W)
    spectrum = np.fft.fft2(img) * np.fft.fft2(padded_k)  # (H, W) complex
    return np.real(np.fft.ifft2(spectrum))  # (H, W)


# ---------------------------------------------------------------------------
# Resampling: bilinear interpolation, pyramids, anti-aliased downsampling
# ---------------------------------------------------------------------------


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
    H, W = img.shape[-2:]
    ys = np.clip(ys, 0.0, H - 1.0)  # (N,)
    xs = np.clip(xs, 0.0, W - 1.0)  # (N,)
    y0 = np.floor(ys).astype(np.int64)  # (N,)
    x0 = np.floor(xs).astype(np.int64)  # (N,)
    y1 = np.minimum(y0 + 1, H - 1)  # (N,)
    x1 = np.minimum(x0 + 1, W - 1)  # (N,)
    a = ys - y0  # (N,) vertical fraction
    b = xs - x0  # (N,) horizontal fraction
    top = (1.0 - b) * img[..., y0, x0] + b * img[..., y0, x1]  # (..., N)
    bottom = (1.0 - b) * img[..., y1, x0] + b * img[..., y1, x1]  # (..., N)
    return (1.0 - a) * top + a * bottom  # (..., N)


def resize_bilinear(img: np.ndarray, out_h: int, out_w: int, align_corners: bool = False) -> np.ndarray:
    """Resize with bilinear interpolation (matches ``F.interpolate(mode="bilinear")``).

    With ``align_corners=False`` (the PyTorch/OpenCV default) source coordinate is
    ``src = (dst + 0.5) · (H_in / H_out) − 0.5``: pixel *centres* are aligned, so the
    image content is not shifted by half a pixel.

    Args:
        img: (H, W) or (C, H, W).
    Returns:
        (out_h, out_w) or (C, out_h, out_w).
    """
    H, W = img.shape[-2:]
    if align_corners:
        ys = np.linspace(0.0, H - 1.0, out_h) if out_h > 1 else np.zeros(1)  # (out_h,)
        xs = np.linspace(0.0, W - 1.0, out_w) if out_w > 1 else np.zeros(1)  # (out_w,)
    else:
        ys = (np.arange(out_h) + 0.5) * (H / out_h) - 0.5  # (out_h,)
        xs = (np.arange(out_w) + 0.5) * (W / out_w) - 0.5  # (out_w,)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")  # each (out_h, out_w)
    flat = bilinear_sample(img, grid_y.ravel(), grid_x.ravel())  # (..., out_h*out_w)
    return flat.reshape(img.shape[:-2] + (out_h, out_w))  # (..., out_h, out_w)


def downsample2(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Anti-aliased ×2 downsampling: Gaussian blur (the low-pass) then keep even pixels.

    Blurring first removes frequencies above the new Nyquist limit, so decimation
    cannot fold them back into low frequencies (aliasing).
    """
    blurred = gaussian_blur(img, sigma)  # (..., H, W)
    return blurred[..., ::2, ::2]  # (..., ceil(H/2), ceil(W/2))


def gaussian_pyramid(img: np.ndarray, levels: int, sigma: float = 1.0) -> list[np.ndarray]:
    """``[G_0 = img, G_1 = down(G_0), …, G_{levels-1}]`` each half the previous size."""
    pyramid = [img.astype(np.float64)]  # level 0: (H, W)
    for _ in range(levels - 1):
        pyramid.append(downsample2(pyramid[-1], sigma))  # (H/2^l, W/2^l)
    return pyramid


def laplacian_pyramid(img: np.ndarray, levels: int, sigma: float = 1.0) -> list[np.ndarray]:
    """Band-pass pyramid ``L_l = G_l − up(G_{l+1})``; the last level is the residual ``G_{levels-1}``.

    Because ``G_l = L_l + up(G_{l+1})``, the image is reconstructed *exactly* by
    :func:`reconstruct_from_laplacian` — it is a lossless, multi-scale representation.
    """
    gauss = gaussian_pyramid(img, levels, sigma)
    lap = []
    for l in range(levels - 1):
        H, W = gauss[l].shape[-2:]
        up = resize_bilinear(gauss[l + 1], H, W)  # (H_l, W_l) upsampled coarse level
        lap.append(gauss[l] - up)  # (H_l, W_l) band-pass detail
    lap.append(gauss[-1])  # (H_{L-1}, W_{L-1}) low-pass residual
    return lap


def reconstruct_from_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    """Invert :func:`laplacian_pyramid` exactly: ``G_l = L_l + up(G_{l+1})`` from the top."""
    current = lap[-1]  # (H_{L-1}, W_{L-1})
    for l in range(len(lap) - 2, -1, -1):
        H, W = lap[l].shape[-2:]
        current = lap[l] + resize_bilinear(current, H, W)  # (H_l, W_l)
    return current


def blur_pool(x: np.ndarray, stride: int = 2, filt_size: int = 3) -> np.ndarray:
    """BlurPool (Zhang, 2019): binomial low-pass filter, then subsample.

    Replaces a plain stride-``s`` subsample with ``blur → stride-s`` so that small input
    shifts do not flip the output (approximate shift-equivariance).

    Args:
        x: (C, H, W) or (H, W).
        filt_size: 2 → [1,1], 3 → [1,2,1], 5 → [1,4,6,4,1] (rows of Pascal's triangle).
    Returns:
        (..., ceil(H/s), ceil(W/s))
    """
    row = np.array([1.0])  # (1,)
    for _ in range(filt_size - 1):
        row = np.convolve(row, np.array([1.0, 1.0]))  # (n+1,) next Pascal row
    row = row / row.sum()  # (filt_size,)
    blurred = separable_filter(x, row, row)  # (..., H, W)
    return blurred[..., ::stride, ::stride]  # (..., ceil(H/s), ceil(W/s))


# ---------------------------------------------------------------------------
# Colour spaces
# ---------------------------------------------------------------------------


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma ``Y = 0.299 R + 0.587 G + 0.114 B``.  ``rgb``: (3, H, W) → (H, W)."""
    weights = np.array([0.299, 0.587, 0.114])  # (3,)
    return np.tensordot(weights, rgb, axes=(0, 0))  # (H, W)


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """BT.601 analog YUV: luma plus two chroma differences (B−Y, R−Y) scaled.

    ``rgb``: (3, H, W) in [0, 1] → (3, H, W) with Y ∈ [0,1], U ∈ ±0.436, V ∈ ±0.615.
    """
    M = np.array(
        [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
    )  # (3, 3)
    return np.tensordot(M, rgb, axes=(1, 0))  # (3, H, W)


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rgb_to_yuv`. ``yuv``: (3, H, W) → (3, H, W)."""
    M = np.array(
        [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
    )  # (3, 3)
    M_inv = np.linalg.inv(M)  # (3, 3)
    return np.tensordot(M_inv, yuv, axes=(1, 0))  # (3, H, W)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Hue ∈ [0, 1), saturation ∈ [0, 1], value ∈ [0, 1] from ``rgb`` (3, H, W) in [0, 1]."""
    r, g, b = rgb[0], rgb[1], rgb[2]  # each (H, W)
    v = rgb.max(axis=0)  # (H, W)
    c = v - rgb.min(axis=0)  # (H, W) chroma
    s = np.where(v > 0, c / np.maximum(v, 1e-12), 0.0)  # (H, W)
    safe_c = np.maximum(c, 1e-12)  # (H, W)
    h = np.where(
        v == r, ((g - b) / safe_c) % 6.0, np.where(v == g, (b - r) / safe_c + 2.0, (r - g) / safe_c + 4.0)
    )  # (H, W) hue in sixths of a turn
    h = np.where(c > 0, h / 6.0, 0.0)  # (H, W)
    return np.stack([h, s, v], axis=0)  # (3, H, W)


def bayer_mosaic(rgb: np.ndarray) -> np.ndarray:
    """Simulate an RGGB Bayer sensor: one colour sample per pixel.

    ``rgb``: (3, H, W) → (H, W) raw mosaic.  Even rows alternate R,G; odd rows G,B.
    Demosaicing (interpolating the two missing channels) is the ISP's first job.
    """
    H, W = rgb.shape[1:]
    raw = np.empty((H, W), dtype=rgb.dtype)  # (H, W)
    raw[0::2, 0::2] = rgb[0, 0::2, 0::2]  # R
    raw[0::2, 1::2] = rgb[1, 0::2, 1::2]  # G
    raw[1::2, 0::2] = rgb[1, 1::2, 0::2]  # G
    raw[1::2, 1::2] = rgb[2, 1::2, 1::2]  # B
    return raw
