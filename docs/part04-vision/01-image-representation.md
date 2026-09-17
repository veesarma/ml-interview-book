# Image representation & signal processing

> **Why this matters at staff level.** Every perception system starts with a sensor, an ISP, a resize and a stack of strided convolutions, and every one of those steps is a sampling operation that can alias, shift, or destroy the signal you are trying to detect. Interviewers use this material two ways: as a coding round ("implement bilinear sampling / a Gaussian pyramid in NumPy") and as a depth probe ("why does a stride-2 conv alias?", "why does ROIAlign use bilinear interpolation?", "why do cameras give you YUV?"). Strong signal is being able to derive the convolution theorem and the Nyquist limit on a whiteboard and then point to exactly where they bite in a real detector.

## TL;DR — the interview card

- An image is a sampled 2-D signal: $I[i, j] = f(i\,\Delta, j\,\Delta)$. Sampling at rate $f_s$ is only lossless if the signal has no energy above $f_s/2$ (**Nyquist**); energy above it *folds back* as aliasing. Downsampling = low-pass **then** decimate. Ever.
- **Convolution** $(f * g)[n] = \sum_k f[k]\, g[n-k]$ flips the kernel; **correlation** does not. `F.conv2d` is correlation. They agree for symmetric kernels (Gaussian, Laplacian) and differ in sign for antisymmetric ones (Sobel).
- **Convolution theorem:** $\mathcal{F}\{f * g\} = \mathcal{F}\{f\}\cdot\mathcal{F}\{g\}$. Blurring is multiplying the spectrum by a low-pass; the Gaussian is its own Fourier transform ($\sigma \leftrightarrow 1/\sigma$), so it is the low-pass with no ringing.
- Gaussian is **separable**: $G(x, y) = g(x) g(y)$, cost $O(HW \cdot 2k)$ instead of $O(HW k^2)$. Sobel is separable too: $[1, 2, 1]^\top [-1, 0, 1]$. Laplacian $\nabla^2 I$ is the 4-neighbour stencil `[[0,1,0],[1,-4,1],[0,1,0]]`; a difference of Gaussians approximates it.
- **Bilinear interpolation** is the tensor product of two linear interpolations; with `align_corners=False` the source coordinate is $\text{src} = (\text{dst} + 0.5)\cdot\text{scale} - 0.5$ so pixel *centres* align. This exact formula is inside `F.interpolate`, `grid_sample`, ROIAlign, and ViT positional-embedding resizing.
- **Gaussian pyramid** $G_{l+1} = \downarrow_2 (G_l * g)$; **Laplacian pyramid** $L_l = G_l - \uparrow(G_{l+1})$ is a lossless band-pass decomposition. FPN is a learned Laplacian pyramid: top-down upsample + lateral add.
- Cameras output **Bayer** mosaics (one colour per photosite, RGGB); the ISP demosaics, white-balances, gamma-encodes, and usually emits **YUV 4:2:0** because human vision (and compression) tolerates chroma at half resolution. Feed the network what it will see in production.
- Stride-2 conv/pooling with no blur aliases; **BlurPool** (blur → stride) restores approximate shift-equivariance and measurably improves both accuracy and consistency under shifts.

## 1. Intuition first

Take a 1-D signal you can see: a cosine with period 2 pixels, $s[n] = \cos(\pi n) = +1, -1, +1, -1, \ldots$. Keep every second sample: you get $+1, +1, +1, \ldots$ — a *constant*. Keep the odd samples instead and you get $-1, -1, -1$. The highest-frequency pattern the original grid could hold has turned into the *lowest*-frequency pattern (DC) on the new grid, and which constant you get depends on the phase. That is aliasing: frequencies above the new Nyquist limit do not disappear, they masquerade as low frequencies. Now blur first with $[\tfrac14, \tfrac12, \tfrac14]$: the cosine becomes $0, 0, 0, \ldots$ (each output averages $+1$ and two $-1$s with those weights: $-\tfrac14 + \tfrac12 - \tfrac14 = 0$). Then decimation returns zeros regardless of phase — the correct answer, "there is no representable content here".

The same story in 2-D, on an image whose frequency grows with radius (a zone plate), is the figure below. Naive stride-4 subsampling produces rings that do not exist in the original (moiré); blurring before each $\times 2$ step does not. The bottom right shows the Laplacian bands: each is the detail the coarser level lost.

![Gaussian pyramid, aliasing from naive decimation vs. anti-aliased downsampling, and Laplacian bands](../assets/figures/part04_pyramid_aliasing.png){ width="720" }

*Top row: a Gaussian pyramid of a zone plate. Bottom left: stride-4 subsampling with no blur creates phantom rings near the edges (aliasing); blur-then-decimate does not. Bottom right: two Laplacian bands, $L_0 = G_0 - \uparrow G_1$ and $L_1$.*

Why should a detection engineer care? Because a ResNet stem is `conv7x7 stride 2 → maxpool stride 2` and each subsequent stage begins with a stride-2 conv. Nothing in a learned $3\times3$ stride-2 kernel forces it to be a low-pass filter, and max-pooling is not linear at all. Shift the input by one pixel and the feature map at stride 32 can change substantially — which is exactly the flakiness you see when a detector's score on a small object oscillates from frame to frame. The remedy (BlurPool) is a five-line change in the stem, and knowing *why* it works is the difference between a senior and a staff answer.

## 2. The math

### 2.1 Sampling and the Nyquist limit

Let $f(x)$ be a continuous signal with Fourier transform $F(\omega)$. Sampling with spacing $\Delta$ multiplies $f$ by an impulse train $\sum_n \delta(x - n\Delta)$. Multiplication in space is convolution in frequency, and the Fourier transform of an impulse train with spacing $\Delta$ is an impulse train with spacing $\omega_s = 2\pi/\Delta$. Therefore the spectrum of the sampled signal is

$$
F_s(\omega) = \frac{1}{\Delta}\sum_{k=-\infty}^{\infty} F(\omega - k\,\omega_s).
$$

The spectrum is *replicated* every $\omega_s$. The replicas do not overlap if and only if $F(\omega) = 0$ for $|\omega| > \omega_s / 2$:

$$
\boxed{\;\text{alias-free sampling} \iff \omega_{\max} < \frac{\omega_s}{2} = \frac{\pi}{\Delta}\;}
$$

That is the Nyquist condition. Decimating by 2 doubles $\Delta$ and halves the allowed bandwidth; any energy in $(\pi/2\Delta, \pi/\Delta)$ folds onto $(0, \pi/2\Delta)$. What it *means*: before you throw away samples you must remove the frequencies the coarser grid cannot represent, and only a low-pass filter can do that.

### 2.2 Convolution, correlation and the convolution theorem

Discrete 2-D convolution and correlation of image $I$ with kernel $h$ (size $k_h \times k_w$, radii $r_h, r_w$):

$$
(I * h)[i, j] = \sum_{u, v} I[i - u, j - v]\, h[u, v], \qquad
(I \star h)[i, j] = \sum_{u, v} I[i + u, j + v]\, h[u, v].
$$

Correlation with $h$ equals convolution with the flipped kernel $h[-u, -v]$. Deep-learning "convolution" layers compute correlation; since the kernel is learned, the distinction is irrelevant *for learning* and matters only when you hand-craft an antisymmetric kernel or reason about the true adjoint (the backward pass of correlation is convolution — see [Convolutions §2](02-convolutions.md)).

The convolution theorem for the DFT: with $\hat{I} = \mathcal{F}\{I\}$ and circular (wrap-around) convolution,

$$
\boxed{\;\mathcal{F}\{I * h\} = \hat{I} \odot \hat{h}\;}
$$

Proof sketch in 1-D: $\mathcal{F}\{I * h\}[\omega] = \sum_n \sum_k I[k] h[n-k] e^{-i\omega n} = \sum_k I[k] e^{-i\omega k} \sum_m h[m] e^{-i\omega m}$ after substituting $m = n - k$. What it means: every linear shift-invariant filter is a *per-frequency gain*. Blurring attenuates high frequencies; sharpening amplifies them; an ideal anti-aliasing filter zeroes everything above the new Nyquist limit. It also gives you the $O(HW\log HW)$ route for large kernels (`fft_convolve2d` below), and it explains why the Gaussian is the preferred blur: $\mathcal{F}\{e^{-x^2/2\sigma^2}\} \propto e^{-\sigma^2\omega^2/2}$ — a Gaussian in frequency too, monotone, no ringing (a box filter's spectrum is a sinc with negative lobes, which is why box-blurred images show faint ghost edges).

### 2.3 Separability

A kernel is separable if $h[u, v] = a[u]\, b[v]$ (rank 1). Then

$$
(I \star h)[i, j] = \sum_u a[u] \Big(\sum_v I[i+u, j+v]\, b[v]\Big),
$$

a horizontal pass followed by a vertical pass: $2k$ multiplies per pixel instead of $k^2$. The 2-D Gaussian is separable by construction ($e^{-(x^2+y^2)/2\sigma^2} = e^{-x^2/2\sigma^2} e^{-y^2/2\sigma^2}$); Sobel is $[1, 2, 1]^\top[-1, 0, 1]$ (a smoothing in one axis, a central difference in the other). Any kernel's SVD tells you how many separable passes approximate it — this is also the idea behind factorised $1\times k$, $k\times1$ convolutions in Inception v3.

### 2.4 Derivative filters

The central difference $\partial I/\partial x \approx (I[i, j+1] - I[i, j-1])/2$ is exact for linear ramps; Sobel adds vertical smoothing so single-pixel noise does not produce spurious edges, and has gain 8 on a unit-slope ramp (the `sobel` test checks exactly this). The Laplacian $\nabla^2 I = I_{xx} + I_{yy}$ with second central differences becomes the stencil $[[0,1,0],[1,-4,1],[0,1,0]]$; on $x^2 + y^2$ it returns exactly 4. Because derivatives amplify high-frequency noise ($\mathcal{F}\{\partial_x\} = i\omega$), practical edge detectors are *derivatives of Gaussians* — blur first, then differentiate, or equivalently convolve with $\partial_x G_\sigma$.

### 2.5 Interpolation

Resampling at a non-integer location $(y, x)$ with $y_0 = \lfloor y \rfloor$, $x_0 = \lfloor x\rfloor$, $a = y - y_0$, $b = x - x_0$:

$$
\boxed{\;I(y, x) = (1-a)(1-b)\, I[y_0, x_0] + (1-a)\, b\, I[y_0, x_0+1] + a(1-b)\, I[y_0+1, x_0] + a\, b\, I[y_0+1, x_0+1]\;}
$$

This is a tensor product of two linear interpolations, so it is *linear in the pixel values* (the four weights depend only on the fractional position) — which is why gradients flow through it to the image (`grid_sample` backward) **and** to the coordinates (spatial transformer networks, deformable convolutions). Nearest neighbour is the same with weights snapped to one corner: cheap, but not differentiable in the coordinates and produces jagged resizes. Bicubic uses a 4×4 neighbourhood with a cubic kernel; it is sharper for photographic upsampling and is what `timm` uses to resize ViT positional embeddings when you change input resolution, because the embedding grid is a smooth 2-D signal that you want to resample without blurring.

The half-pixel convention matters. If you map destination index $d$ to source $d \cdot s$ (with $s = H_{\text{in}}/H_{\text{out}}$) you align the *top-left corners* and shift all content by $(s-1)/2$ pixels. Aligning *centres* gives

$$
\text{src} = (d + 0.5)\, s - 0.5,
$$

which is `align_corners=False` in PyTorch and the OpenCV default. `align_corners=True` maps $0 \to 0$ and $H_{\text{out}}-1 \to H_{\text{in}}-1$; it is only sensible when the samples are corner-anchored (some segmentation code paths). A mismatch here is a classic source of a half-pixel systematic error in box regression targets or in mask reprojection.

### 2.6 Pyramids

Gaussian pyramid: $G_0 = I$, $G_{l+1} = (G_l * g_\sigma)\!\downarrow_2$. Laplacian pyramid: $L_l = G_l - \uparrow\!(G_{l+1})$ for $l < L-1$ and $L_{L-1} = G_{L-1}$. Reconstruction is exact by construction: $G_l = L_l + \uparrow\!(G_{l+1})$, applied from the top. The bands are approximately octave-wide band-pass filters (difference of Gaussians $\approx$ scale-normalised Laplacian), which is why blob detectors (SIFT/LoG) look for extrema across pyramid levels.

Feature Pyramid Networks reuse this shape: bottom-up backbone features $C_3 \ldots C_5$ play the role of $G_l$, the top-down path upsamples and *adds* a lateral $1\times1$ projection, so $P_l$ carries both the semantics of the coarse level and the localisation of the fine level. The add (not concat) and the nearest-neighbour upsample are cheap by design; the $3\times3$ smoothing conv after each add exists precisely to undo the aliasing of nearest-neighbour upsampling.

### 2.7 Colour

A sensor photosite counts photons through one colour filter. The Bayer pattern (RGGB) gives half the sites green because luminance acuity is what humans notice and the green filter sits at the peak of the eye's response. Demosaicing interpolates the missing two channels per pixel (an interpolation problem again, with all its aliasing risks — colour moiré on fine textures). The ISP then applies black-level subtraction, white balance, a colour matrix, gamma ($\approx x^{1/2.2}$, so that quantisation to 8 bits spends more codes on dark tones), and converts to YUV/YCbCr:

$$
Y = 0.299R + 0.587G + 0.114B, \qquad U \propto B - Y, \qquad V \propto R - Y.
$$

Chroma is then subsampled 4:2:0 (half resolution in both axes), because compression and display pipelines exploit that our chroma acuity is low. Hardware decoders and camera pipelines therefore hand you NV12/YUV420, and a production network that consumes YUV planes directly saves a colour conversion per frame. HSV separates hue from lightness (handy for colour-jitter augmentation and for hand-written thresholds); CIELAB is perceptually uniform (equal Euclidean steps ≈ equal perceived difference), which is why colour-difference metrics and some photometric losses use it.

## 3. Implementation

All of this is in `src/mlbook/vision/image_ops.py`. The core is a correlation routine that loops over kernel *taps*, not pixels, so each of the $k_h k_w$ iterations is a vectorised shifted-window multiply-add:

```python
def correlate2d(img, kernel, pad_mode="reflect"):
    k_h, k_w = kernel.shape
    r_h, r_w = k_h // 2, k_w // 2
    padded = pad2d(img, r_h, r_w, pad_mode)          # (..., H+2r_h, W+2r_w)
    H, W = img.shape[-2:]
    out = np.zeros_like(img, dtype=np.float64)       # (..., H, W)
    for u in range(k_h):
        for v in range(k_w):
            window = padded[..., u : u + H, v : v + W]   # (..., H, W) shifted view
            out += kernel[u, v] * window
    return out

def convolve2d(img, kernel, pad_mode="reflect"):
    return correlate2d(img, kernel[::-1, ::-1], pad_mode)  # flip → true convolution
```

`pad_mode="reflect"` mirrors the border without repeating the edge pixel (NumPy's "reflect" equals SciPy's "mirror" — the test pins this down, because the two libraries use the word for different things). Separable filtering is two calls with $1\times k$ and $k \times 1$ kernels, and the Gaussian builds on it:

```python
def separable_filter(img, k_col, k_row, pad_mode="reflect"):
    tmp = correlate2d(img, k_row[None, :], pad_mode)   # (..., H, W) horizontal pass
    return correlate2d(tmp, k_col[:, None], pad_mode)  # (..., H, W) vertical pass

def gaussian_blur(img, sigma, pad_mode="reflect"):
    g = gaussian_kernel_1d(sigma)                       # (2r+1,), r = ceil(3σ), sums to 1
    return separable_filter(img, g, g, pad_mode)        # (..., H, W)
```

Bilinear sampling gathers the four neighbours with fancy indexing, so it works for any number of query points and for `(C, H, W)` images at once; `resize_bilinear` builds the half-pixel-aligned grid on top of it:

```python
def bilinear_sample(img, ys, xs):
    H, W = img.shape[-2:]
    ys = np.clip(ys, 0.0, H - 1.0); xs = np.clip(xs, 0.0, W - 1.0)   # (N,) border clamp
    y0 = np.floor(ys).astype(np.int64); x0 = np.floor(xs).astype(np.int64)  # (N,)
    y1 = np.minimum(y0 + 1, H - 1);     x1 = np.minimum(x0 + 1, W - 1)      # (N,)
    a = ys - y0; b = xs - x0                                                 # (N,) fractions
    top    = (1 - b) * img[..., y0, x0] + b * img[..., y0, x1]   # (..., N)
    bottom = (1 - b) * img[..., y1, x0] + b * img[..., y1, x1]   # (..., N)
    return (1 - a) * top + a * bottom                            # (..., N)

def resize_bilinear(img, out_h, out_w, align_corners=False):
    H, W = img.shape[-2:]
    if align_corners:
        ys = np.linspace(0.0, H - 1.0, out_h); xs = np.linspace(0.0, W - 1.0, out_w)
    else:
        ys = (np.arange(out_h) + 0.5) * (H / out_h) - 0.5   # (out_h,) centre-aligned
        xs = (np.arange(out_w) + 0.5) * (W / out_w) - 0.5   # (out_w,)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")     # (out_h, out_w) each
    flat = bilinear_sample(img, grid_y.ravel(), grid_x.ravel())   # (..., out_h*out_w)
    return flat.reshape(img.shape[:-2] + (out_h, out_w))
```

The pyramid functions compose these: `downsample2` = blur then `[::2, ::2]`; `laplacian_pyramid` subtracts the bilinearly upsampled next level; `reconstruct_from_laplacian` adds it back. `blur_pool` builds the binomial filter from Pascal's triangle and strides after blurring. `fft_convolve2d` places the kernel at the origin with `np.roll` and multiplies spectra — the convolution theorem as five lines.

**How you'd test it.** `gaussian_blur` against `scipy.ndimage.gaussian_filter` (same truncation, matching border mode); separable vs. full 2-D kernel; `correlate2d` vs. `F.conv2d`; an impulse image to show `convolve2d` returns the kernel while `correlate2d` returns it flipped; `resize_bilinear` vs. `F.interpolate` for both `align_corners` settings; exact Laplacian reconstruction; the Nyquist-stripes example (naive decimation gives a phase-dependent constant, blur-then-decimate gives 0.5); BlurPool's output changes less under a one-pixel shift than a strided subsample does. Run `pytest tests/test_vision_image_ops.py -q`.

??? example "Full implementation — `src/mlbook/vision/image_ops.py`"
    ```python
    --8<-- "src/mlbook/vision/image_ops.py"
    ```

## Retype by hand

| Symbol (file `src/mlbook/vision/image_ops.py`) | Reproduce from memory? | Test |
|---|---|---|
| `correlate2d`, `convolve2d` | **Yes** — the tap-loop formulation is the canonical "implement a 2-D filter" answer | `test_correlate_matches_torch_conv2d`, `test_convolution_flips_kernel` |
| `gaussian_kernel_1d`, `separable_filter`, `gaussian_blur` | **Yes** | `test_gaussian_blur_matches_scipy`, `test_separable_equals_full_2d_kernel` |
| `bilinear_sample`, `resize_bilinear` | **Yes** — asked verbatim in coding rounds and inside ROIAlign | `test_bilinear_sample_exact_at_integer_and_midpoint`, `test_bilinear_resize_matches_torch` |
| `downsample2`, `gaussian_pyramid`, `laplacian_pyramid`, `reconstruct_from_laplacian` | **Yes** (short once the above exist) | `test_laplacian_pyramid_reconstructs_exactly`, `test_downsample_removes_nyquist_stripes` |
| `blur_pool` | Read; be able to explain | `test_blur_pool_is_more_shift_invariant_than_strided_subsample` |
| `sobel`, `laplacian`, `fft_convolve2d` | Read; know the stencils | `test_sobel_sign_on_ramp`, `test_laplacian_of_quadratic_is_constant`, `test_fft_convolution_theorem` |
| `rgb_to_gray`, `rgb_to_yuv`, `rgb_to_hsv`, `bayer_mosaic` | Read | `test_colour_roundtrip_and_hsv` |

Check: `pytest tests/test_vision_image_ops.py -q`. Target time: **filtering + separable Gaussian: 15 minutes; bilinear sampling + resize: 15 minutes; pyramid: 10 minutes.**

## 4. Systems view: cost, failure modes, trade-offs

**Cost.** A direct $k\times k$ filter is $HWk^2$ MACs; separable is $2HWk$; FFT is $O(HW\log HW)$ regardless of $k$ but with large constants and padding overhead, so it only wins for $k \gtrsim 15$. On a GPU the $3\times3$ correlation is a tiny fraction of a network's cost; the real costs of this chapter are *bandwidth* (every resize reads and writes the full image; an ISP pipeline on a 4K 30 fps stream is a memory-bound problem) and *latency* placement (resizing on the CPU before H2D copy vs. on the GPU after).

**Failure modes.**

| Symptom | Root cause | Fix |
|---|---|---|
| Detector score flickers between adjacent frames | strided conv/pool without low-pass; features not shift-equivariant | BlurPool in stem and stage transitions; test-time augmentation with shifts to measure |
| Small text/objects vanish after resize | resize is nearest or unfiltered bilinear at large scale factors (bilinear only averages 2×2 — for a 4× reduction it aliases) | area interpolation (`cv2.INTER_AREA`) or Gaussian pre-blur; keep native resolution for small-object heads |
| Half-pixel systematic error in boxes or masks | corner-aligned vs centre-aligned resampling mismatch between training and serving | fix one convention (`align_corners=False`) end to end; unit-test it |
| Colour moiré on fabrics, false edges | demosaicing aliasing; JPEG chroma subsampling | train on the production ISP output, not on PNG stills |
| Model works on RGB frames, fails on the device | device delivers YUV420 / different gamma / different white balance | feed the network the same colour space and tone curve it will see; augment with ISP variation |

**When to use what.**

| Need | Use | Rule |
|---|---|---|
| Shrink an image by a large factor | area / Gaussian-then-decimate | if scale factor > 2, bilinear alone aliases |
| Enlarge for display or ViT positional embeddings | bicubic | smooth signals: cubic is sharper without ringing artefacts of Lanczos |
| Differentiable warp (flow, ROIAlign, STN) | bilinear | gradients wrt coordinates exist and are cheap |
| Masks / label maps | nearest | interpolating class IDs is meaningless |
| Anti-alias inside a CNN | BlurPool (binomial 3 or 5) | cheap, measurable robustness gain |

## 5. In production

!!! production "Tesla — raw-ish photon counts instead of ISP output"
    At Tesla AI Day (2021) the Autopilot vision team described moving away from ISP-processed images toward feeding the network *photon-count* data with minimal ISP processing (the "raw" 12-bit sensor data), the argument being that the ISP is tuned for human viewing (tone mapping, noise reduction) and discards dynamic range the network can use, especially at night. The trade-off: more bandwidth per frame and a network that must learn white balance and tone mapping itself, in exchange for low-light range. Source: Tesla AI Day 2021 presentation (recorded talk; search "Tesla AI Day 2021 vision"). This is a design choice reported in a talk, not a paper.

!!! production "Adobe / UC Berkeley — BlurPool for shift-invariant CNNs"
    Zhang's "Making Convolutional Networks Shift-Invariant Again" (ICML 2019, arXiv:1904.11486) showed that inserting a binomial low-pass filter before every stride-2 operation in ResNet/DenseNet/MobileNet improved ImageNet accuracy and greatly improved classification consistency under small input shifts. The alternative it rejects — hoping data augmentation teaches invariance — does not fix the aliasing mechanism and costs training data; BlurPool fixes the mechanism for a few percent extra compute. `blur_pool` above is the reference implementation.

!!! production "NVIDIA — hardware ISP → YUV → network on DRIVE and Jetson"
    NVIDIA's DRIVE and Jetson platforms run camera capture through a hardware ISP and deliver frames in YUV formats (NV12) to the inference pipeline; DeepStream and TensorRT preprocessing plugins convert or feed planar YUV directly. The engineering reason is bandwidth and latency: colour conversion on a 4K, multi-camera stream is a memory-bound kernel you would rather not run, and the video encoder for logging already wants 4:2:0. Source: NVIDIA DeepStream SDK documentation (search "DeepStream NV12 preprocessing"). The exact preprocessing choices are product-specific; treat the YUV-in pattern as the general lesson.

!!! production "Google — FPN as the pyramid everyone ships"
    Lin et al.'s Feature Pyramid Networks (CVPR 2017, arXiv:1612.03144) formalised the Laplacian-pyramid idea as a learned top-down path and became the default neck in Detectron/Detectron2, EfficientDet (as BiFPN) and every YOLO after v3. The trade-off they chose over image pyramids (running the backbone at several scales) was compute: one backbone pass plus a cheap neck gives multi-scale features at roughly the cost of a single-scale detector.

## 6. Interview questions and strong answers

!!! interview "Why does a stride-2 convolution alias, and what would you do about it?"
    A learned $3\times3$ stride-2 kernel is a filter followed by decimation, but nothing constrains the filter to be low-pass; and max-pooling is nonlinear and only worsens it. Frequencies above the new Nyquist limit fold back, so a one-pixel shift of the input can change the stride-32 feature map non-trivially — you see it as score flicker on small objects. Fix: put a fixed binomial blur before each stride (BlurPool), or use anti-aliased downsampling in the data pipeline; measure with a shift-consistency metric.
    **Staff follow-up:** *Does this matter for ViTs?* The patch-embedding conv is a stride-16 conv with a $16\times16$ kernel — the kernel covers the whole stride, so it *can* be low-pass, but the learned kernel usually is not, and ViTs are known to be sensitive to sub-patch shifts. Positional-embedding interpolation at a new resolution is a second resampling step with the same concerns.

!!! interview "Convolution versus correlation — does it matter for a CNN?"
    Correlation is convolution with a flipped kernel. For a *learned* kernel the flip is absorbed into the parameters, so `F.conv2d` computing correlation is harmless. It matters when (a) you hand-craft antisymmetric kernels (Sobel sign flips), (b) you reason about adjoints — the backward pass of correlation w.r.t. the input is a true convolution with the same kernel (transposed conv), and (c) you use the FFT route, where the theorem is stated for convolution.
    **Staff follow-up:** *Prove that the input-gradient of a stride-1 correlation is a convolution.* $\partial L/\partial x[p] = \sum_q \partial L/\partial y[q]\, w[p - q]$ — the kernel index is reversed relative to the forward $y[q] = \sum_p x[p]\, w[p - q]$, so it is a convolution of the upstream gradient with $w$.

!!! interview "Why does ROIAlign use bilinear interpolation and why did it matter so much for masks?"
    ROIPool quantises the box to integer feature cells twice (box edges and bin edges), a misalignment of up to half a stride (16 px at stride 32) between the crop and the object. For classification that is noise; for a $28\times28$ mask that is a systematic offset of several mask pixels. Bilinear sampling at exact continuous positions removes the quantisation and is differentiable, so the mask head trains cleanly. Mask R-CNN reports large mask-AP gains from this one change.
    **Staff follow-up:** *What does ROIAlign do at the box boundary and why does `sampling_ratio` exist?* Each bin averages $r \times r$ bilinear samples at fixed interior positions; $r=2$ is a compromise between aliasing (a single sample per bin ignores most of the bin) and cost.

!!! interview "You are handed NV12 frames from the camera pipeline. Do you convert to RGB before the network?"
    Prefer not to: convert the *training* data to match the serving format instead. YUV420 has luma at full resolution and chroma at half, which is what the encoder already produces; a network can take Y at full res and UV at half res as two inputs (or you can upsample UV cheaply). Converting to RGB costs a memory-bound pass per frame per camera and can introduce a gamma/range mismatch (limited vs full range) that silently shifts the input distribution.
    **Staff follow-up:** *What breaks when the ISP firmware is updated?* Tone curve, denoising and sharpening change the input distribution; treat the ISP version as a feature of the dataset and gate deployments on a distribution-shift check.

!!! interview "Explain the Laplacian pyramid and where it appears in modern detectors."
    $L_l = G_l - \uparrow G_{l+1}$: each level holds one octave of detail; the pyramid is lossless and the bands are near-orthogonal. FPN is the learned analogue — coarse semantics upsampled and *added* to fine lateral features, with a $3\times3$ conv to clean the upsampling artefacts. The difference is that FPN adds instead of subtracts, because it wants each level to contain everything at and above its scale rather than a single band.
    **Staff follow-up:** *Why add rather than concatenate?* Channel count stays constant across levels so one shared head can run on all of them, which is what makes RetinaNet/FCOS heads weight-shared across scales.

!!! interview "Derive the half-pixel-centre formula for resizing."
    Pixel $d$ of the output covers the interval $[d, d+1)$ in output units, whose centre is $d + 0.5$; scaling to input units gives $(d + 0.5)\, s$; the input pixel whose centre sits there has index $(d + 0.5)\, s - 0.5$. Aligning corners instead ($d \cdot s$) shifts every output by $(s - 1)/2$ pixels — for a $4\times$ downsample, 1.5 input pixels — which shows up as a constant bias in box regression targets when the resize and the anchor grid disagree.

## 7. Exercises

1. ★ Show that the box filter $[1, 1, 1]/3$ has negative lobes in its spectrum (compute $\hat{h}(\omega) = (1 + 2\cos\omega)/3$) and find the first frequency where it goes negative. Explain what that does to a downsampled image.

    ??? success "Solution"
        $\hat{h}(\omega) = (1 + 2\cos\omega)/3 < 0$ when $\cos\omega < -1/2$, i.e. $\omega > 2\pi/3$. Frequencies in $(2\pi/3, \pi)$ are passed with *inverted sign* — not removed — so after decimation they alias with flipped contrast: faint reversed-contrast ghosts near sharp edges. The Gaussian's spectrum is positive everywhere, which is why it is preferred.

2. ★★ (coding) Implement `gaussian_blur_fft(img, sigma)` using `fft_convolve2d` and show it matches `gaussian_blur` with `pad_mode="wrap"` to $10^{-10}$. Then time both for $\sigma = 1, 4, 16$ on a $512\times512$ image and report the crossover.

    ??? success "Solution"
        ```python
        from mlbook.vision.image_ops import gaussian_kernel_2d, fft_convolve2d, gaussian_blur
        def gaussian_blur_fft(img, sigma):
            return fft_convolve2d(img, gaussian_kernel_2d(sigma))
        img = np.random.rand(512, 512)
        for s in (1, 4, 16):
            assert np.allclose(gaussian_blur_fft(img, s), gaussian_blur(img, s, pad_mode="wrap"), atol=1e-10)
        ```
        The separable direct filter costs $2 \cdot 512^2 \cdot (6\sigma + 1)$ MACs; the FFT route costs two forward and one inverse FFT regardless of $\sigma$. On a laptop the crossover is around $\sigma \approx 8$–$16$ (kernel side ≈ 50–100); below that the separable filter wins.

3. ★★ Prove that bilinear interpolation reproduces any function of the form $f(y, x) = \alpha + \beta y + \gamma x + \delta\, x y$ exactly, and give a function it does not reproduce. Why does this make ROIAlign exact on linear ramps (the `roi_align` test)?

    ??? success "Solution"
        The four-corner formula is the unique bilinear (degree ≤ 1 in each variable separately) interpolant of the corner values; the space of such functions is spanned by $\{1, y, x, xy\}$, so any member is reproduced. $f = x^2$ is not: at the midpoint bilinear returns $(0 + 1)/2 = 0.5$ vs the true $0.25$. A feature map linear in $x$ is therefore sampled exactly at any continuous position, so ROIAlign's bin averages equal the ramp at the bin centres.

4. ★★★ (coding) Build a 5-level Laplacian pyramid of two images and blend them with a Gaussian-pyramid mask (Burt–Adelson multiresolution blending): $L^{\text{out}}_l = M_l L^A_l + (1 - M_l) L^B_l$, then reconstruct. Explain in two sentences why blending per band avoids the visible seam that a single-scale alpha blend produces.

    ??? success "Solution"
        ```python
        from mlbook.vision.image_ops import laplacian_pyramid, gaussian_pyramid, reconstruct_from_laplacian
        LA, LB = laplacian_pyramid(A, 5), laplacian_pyramid(B, 5)
        M = gaussian_pyramid(mask.astype(float), 5)
        out = reconstruct_from_laplacian([m * a + (1 - m) * b for m, a, b in zip(M, LA, LB)])
        ```
        A hard mask blended at full resolution transitions over one pixel for *all* frequencies, so low-frequency content (illumination) jumps visibly. Blending each band with a mask blurred to that band's scale makes the transition width proportional to the wavelength: fine detail switches over a few pixels, coarse tone over many, and no single seam is visible.

5. ★★★ A 4K (3840×2160) 30 fps camera feeds a detector that runs at 960×540. Compare three pipelines — (a) bilinear resize on the CPU, (b) area resize on the CPU, (c) H2D copy of the full frame then GPU resize — in terms of aliasing, CPU cost and PCIe bandwidth, and choose one for an 8-camera vehicle. State your assumptions.

    ??? success "Solution"
        (a) is a $4\times$ reduction with a 2×2 footprint: it aliases and drops thin structures (lane markings, distant poles). (b) integrates over the full $4\times4$ footprint (a box low-pass) — correct anti-aliasing, roughly $2\times$ the CPU cost of (a). (c) moves 8 × 3840×2160 × 1.5 bytes (NV12) × 30 fps ≈ 3 GB/s over PCIe *before* any compute, which is a large fraction of a Gen3 x4 link but fine for x16; the GPU then resizes with a proper filter in a memory-bound kernel that is cheap relative to the detector. For a vehicle you would choose (c) if the cameras are attached to the accelerator (as on DRIVE/Jetson-class SoCs, where capture, ISP and inference share memory and the "copy" is free), else (b) on a dedicated preprocessing core. Never (a) at $4\times$.

## References

Links could not be verified from this build environment, so titles, venues and arXiv IDs are given for you to search.

- R. Zhang, "Making Convolutional Networks Shift-Invariant Again", ICML 2019, arXiv:1904.11486.
- T.-Y. Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017, arXiv:1612.03144.
- K. He et al., "Mask R-CNN", ICCV 2017, arXiv:1703.06870 (ROIAlign, §3).
- P. Burt and E. Adelson, "The Laplacian Pyramid as a Compact Image Code", IEEE Trans. Communications, 1983.
- A. Oppenheim and R. Schafer, *Discrete-Time Signal Processing* — sampling theorem and the DFT convolution theorem.
- R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. — chapters on image processing and pyramids.
- Tesla AI Day 2021 (recorded presentation) — the vision stack discussion of raw photon counts and multi-camera fusion.
- NVIDIA DeepStream SDK documentation — NV12 input and preprocessing plugins.
