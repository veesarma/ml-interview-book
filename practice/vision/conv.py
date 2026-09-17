# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/vision/conv.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k conv -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py vision/conv --force

"""A convolution layer from scratch (NumPy): naive loops, im2col/GEMM, and backward.

Shapes follow PyTorch: input ``x`` is ``(B, C_in, H, W)``, weight ``w`` is
``(C_out, C_in // groups, k_h, k_w)``, output is ``(B, C_out, H_out, W_out)`` with

    H_out = floor((H + 2 p_h - d_h (k_h - 1) - 1) / s_h) + 1.

Every function here is what ``torch.nn.functional.conv2d`` computes (cross-correlation).
"""
from __future__ import annotations
import numpy as np

def conv_output_size(n: int, k: int, stride: int=1, pad: int=0, dilation: int=1) -> int:
    """``floor((n + 2p − d(k−1) − 1) / s) + 1`` along one spatial axis."""
    raise NotImplementedError('TODO: implement conv_output_size (see the reference in src/mlbook)')

def receptive_field(layers: list[tuple[int, int, int]]) -> tuple[int, int]:
    """Theoretical receptive field of a stack of (kernel, stride, dilation) layers.

    Recursion (``j`` = jump = product of strides so far, ``r`` = receptive field):
        r_{l} = r_{l-1} + (d_l (k_l − 1)) · j_{l-1},   j_l = j_{l-1} · s_l.
    Returns:
        (r, j) after the last layer, in input pixels.
    """
    raise NotImplementedError('TODO: implement receptive_field (see the reference in src/mlbook)')

def conv_flops_and_params(c_in: int, c_out: int, k: int, h_out: int, w_out: int, groups: int=1, bias: bool=True) -> tuple[int, int]:
    """Multiply-adds and parameters of a ``k×k`` conv producing ``(c_out, h_out, w_out)``.

    MACs   = h_out · w_out · c_out · (c_in / groups) · k²
    params = c_out · (c_in / groups) · k²  (+ c_out for bias)
    Depthwise (groups = c_in = c_out) divides both by c_in.
    """
    raise NotImplementedError('TODO: implement conv_flops_and_params (see the reference in src/mlbook)')

def depthwise_separable_flops(c_in: int, c_out: int, k: int, h_out: int, w_out: int) -> tuple[int, int]:
    """MACs of ``k×k`` depthwise + ``1×1`` pointwise, and of the equivalent standard conv.

    Ratio ≈ 1/c_out + 1/k²  (MobileNet v1, eq. 5).
    """
    raise NotImplementedError('TODO: implement depthwise_separable_flops (see the reference in src/mlbook)')

def conv2d_naive(x: np.ndarray, w: np.ndarray, stride: int=1, pad: int=0, dilation: int=1) -> np.ndarray:
    """Cross-correlation by explicit loops (groups=1).  For understanding, not speed.

    ``out[b, o, i, j] = Σ_{c,u,v} x_pad[b, c, i·s + u·d, j·s + v·d] · w[o, c, u, v]``

    Args:
        x: (B, C_in, H, W).  w: (C_out, C_in, k_h, k_w).
    Returns:
        (B, C_out, H_out, W_out).
    """
    raise NotImplementedError('TODO: implement conv2d_naive (see the reference in src/mlbook)')

def _tap_indices(H_out: int, W_out: int, k_h: int, k_w: int, stride: int, dilation: int):
    """Row/col indices of every (output position, kernel tap) pair.

    Returns:
        rows, cols: each (k_h·k_w, H_out·W_out) int arrays into the padded input.
    """
    raise NotImplementedError('TODO: implement _tap_indices (see the reference in src/mlbook)')

def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int=1, pad: int=0, dilation: int=1) -> np.ndarray:
    """Unfold every receptive field into a column (``torch.nn.Unfold``).

    Args:
        x: (B, C, H, W).
    Returns:
        (B, C·k_h·k_w, H_out·W_out) — column ``p`` holds the patch under output position ``p``,
        ordered ``(c, u, v)`` to match ``w.reshape(C_out, -1)``.
    """
    raise NotImplementedError('TODO: implement im2col (see the reference in src/mlbook)')

def col2im(cols: np.ndarray, x_shape: tuple[int, int, int, int], k_h: int, k_w: int, stride: int, pad: int, dilation: int) -> np.ndarray:
    """Adjoint of :func:`im2col`: scatter-*add* columns back to image positions (``torch.nn.Fold``).

    Overlapping patches accumulate, which is exactly what the gradient w.r.t. the input needs.

    Args:
        cols: (B, C·k_h·k_w, H_out·W_out).
    Returns:
        (B, C, H, W) — padding rows/cols are cropped away.
    """
    raise NotImplementedError('TODO: implement col2im (see the reference in src/mlbook)')

def conv2d_im2col(x: np.ndarray, w: np.ndarray, b: np.ndarray | None=None, stride: int=1, pad: int=0, dilation: int=1, groups: int=1) -> tuple[np.ndarray, tuple]:
    """Forward pass as a GEMM per group: ``out_g = W_g @ cols_g``.

    Args:
        x: (B, C_in, H, W).  w: (C_out, C_in/groups, k_h, k_w).  b: (C_out,) or None.
    Returns:
        out: (B, C_out, H_out, W_out) and a cache for :func:`conv2d_backward`.
    """
    raise NotImplementedError('TODO: implement conv2d_im2col (see the reference in src/mlbook)')

def conv2d_backward(dout: np.ndarray, w: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients of ``L`` w.r.t. input, weight and bias given ``dout = ∂L/∂out``.

    With ``out = W @ cols``:  ``∂L/∂W = dout @ colsᵀ``,  ``∂L/∂cols = Wᵀ @ dout``,
    then ``∂L/∂x = col2im(∂L/∂cols)`` (the scatter-add is the adjoint of the gather).

    Args:
        dout: (B, C_out, H_out, W_out).
    Returns:
        dx (B, C_in, H, W), dw (C_out, C_in/groups, k_h, k_w), db (C_out,).
    """
    raise NotImplementedError('TODO: implement conv2d_backward (see the reference in src/mlbook)')

def conv_transpose2d(x: np.ndarray, w: np.ndarray, stride: int=1, pad: int=0) -> np.ndarray:
    """Transposed convolution = the input-gradient of a conv with the same weights.

    Args:
        x: (B, C_in, H, W) — plays the role of ``dout`` of the forward conv.
        w: (C_in, C_out, k, k) — PyTorch's ``ConvTranspose2d`` weight layout.
    Returns:
        (B, C_out, (H−1)·s − 2p + k, (W−1)·s − 2p + k).
    """
    raise NotImplementedError('TODO: implement conv_transpose2d (see the reference in src/mlbook)')
