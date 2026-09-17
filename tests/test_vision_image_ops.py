import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from mlbook.vision import image_ops as ops


def test_gaussian_blur_matches_scipy():
    img = np.random.rand(20, 24)
    ours = ops.gaussian_blur(img, sigma=1.5)  # (20, 24)
    ref = ndi.gaussian_filter(img, sigma=1.5, mode="mirror", truncate=3.0)  # np.pad "reflect" == scipy "mirror"
    assert np.allclose(ours, ref, atol=1e-6)


def test_separable_equals_full_2d_kernel():
    img = np.random.rand(3, 16, 17)
    g = ops.gaussian_kernel_1d(1.0)
    full = ops.correlate2d(img, np.outer(g, g))
    sep = ops.separable_filter(img, g, g)
    assert np.allclose(full, sep)


def test_correlate_matches_torch_conv2d():
    img = np.random.rand(12, 13)
    k = np.random.rand(3, 5)
    ours = ops.correlate2d(img, k, pad_mode="constant")
    ref = F.conv2d(torch.tensor(img)[None, None], torch.tensor(k)[None, None], padding=(1, 2))[0, 0].numpy()
    assert np.allclose(ours, ref)


def test_convolution_flips_kernel():
    img = np.zeros((7, 7))
    img[3, 3] = 1.0  # impulse: convolution returns the kernel, correlation returns it flipped
    k = np.arange(9.0).reshape(3, 3)
    assert np.allclose(ops.convolve2d(img, k)[2:5, 2:5], k)
    assert np.allclose(ops.correlate2d(img, k)[2:5, 2:5], k[::-1, ::-1])


def test_sobel_sign_on_ramp():
    ramp = np.tile(np.arange(10.0), (10, 1))  # intensity rises to the right
    gx, gy, mag = ops.sobel(ramp)
    assert np.all(gx[2:-2, 2:-2] > 0)
    assert np.allclose(gy[2:-2, 2:-2], 0.0)
    assert np.allclose(gx[2:-2, 2:-2], 8.0)  # Sobel gain is 8 on a unit-slope ramp


def test_laplacian_of_quadratic_is_constant():
    yy, xx = np.mgrid[0:12, 0:12].astype(float)
    quad = xx**2 + yy**2  # ∇² = 4
    assert np.allclose(ops.laplacian(quad)[2:-2, 2:-2], 4.0)


def test_fft_convolution_theorem():
    img = np.random.rand(16, 16)
    k = np.random.rand(5, 3)
    direct = ops.convolve2d(img, k, pad_mode="wrap")  # circular convolution
    assert np.allclose(ops.fft_convolve2d(img, k), direct, atol=1e-10)


def test_bilinear_resize_matches_torch():
    img = np.random.rand(2, 9, 11)
    for align in (False, True):
        ours = ops.resize_bilinear(img, 14, 6, align_corners=align)
        ref = F.interpolate(torch.tensor(img)[None], size=(14, 6), mode="bilinear", align_corners=align)[0].numpy()
        assert np.allclose(ours, ref, atol=1e-6), align


def test_bilinear_sample_exact_at_integer_and_midpoint():
    img = np.arange(16.0).reshape(4, 4)
    assert ops.bilinear_sample(img, np.array([2.0]), np.array([3.0]))[0] == 11.0
    assert ops.bilinear_sample(img, np.array([1.5]), np.array([1.5]))[0] == (5 + 6 + 9 + 10) / 4


def test_laplacian_pyramid_reconstructs_exactly():
    img = np.random.rand(32, 48)
    lap = ops.laplacian_pyramid(img, levels=4)
    assert [l.shape for l in lap] == [(32, 48), (16, 24), (8, 12), (4, 6)]
    assert np.allclose(ops.reconstruct_from_laplacian(lap), img, atol=1e-12)


def test_downsample_removes_nyquist_stripes():
    # 1-pixel stripes sit exactly at Nyquist; naive decimation keeps a constant (aliased)
    # image whose value depends on the phase; blurring first averages the stripes away.
    stripes = np.tile(np.array([0.0, 1.0]), (16, 8))  # (16, 16)
    naive = stripes[::2, ::2]
    assert naive.std() == 0.0 and naive.mean() == 0.0  # phase-dependent: all zeros here
    aa = ops.downsample2(stripes, sigma=1.0)
    assert abs(aa.mean() - 0.5) < 0.05  # correct low-pass answer: mid-gray


def test_blur_pool_is_more_shift_invariant_than_strided_subsample():
    x = np.zeros((1, 32, 32))
    x[0, 10:20, 8:24] = 1.0
    shifted = np.roll(x, 1, axis=2)
    naive_diff = np.abs(x[..., ::2, ::2] - shifted[..., ::2, ::2]).sum()
    bp_diff = np.abs(ops.blur_pool(x) - ops.blur_pool(shifted)).sum()
    assert bp_diff < naive_diff


def test_colour_roundtrip_and_hsv():
    rgb = np.random.rand(3, 5, 6)
    assert np.allclose(ops.yuv_to_rgb(ops.rgb_to_yuv(rgb)), rgb)
    pure_red = np.zeros((3, 1, 1)); pure_red[0] = 1.0
    h, s, v = ops.rgb_to_hsv(pure_red)[:, 0, 0]
    assert (h, s, v) == (0.0, 1.0, 1.0)
    gray = np.full((3, 2, 2), 0.3)
    assert np.allclose(ops.rgb_to_hsv(gray)[1], 0.0)  # zero saturation
    raw = ops.bayer_mosaic(rgb)
    assert raw.shape == (5, 6) and raw[0, 0] == rgb[0, 0, 0] and raw[1, 1] == rgb[2, 1, 1]
