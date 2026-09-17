"""A convolution layer from scratch (NumPy): naive loops, im2col/GEMM, and backward.

Shapes follow PyTorch: input ``x`` is ``(B, C_in, H, W)``, weight ``w`` is
``(C_out, C_in // groups, k_h, k_w)``, output is ``(B, C_out, H_out, W_out)`` with

    H_out = floor((H + 2 p_h - d_h (k_h - 1) - 1) / s_h) + 1.

Every function here is what ``torch.nn.functional.conv2d`` computes (cross-correlation).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Arithmetic: output size, receptive field, FLOPs, parameters
# ---------------------------------------------------------------------------


def conv_output_size(n: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> int:
    """``floor((n + 2p − d(k−1) − 1) / s) + 1`` along one spatial axis."""
    effective_k = dilation * (k - 1) + 1  # the span the dilated kernel covers
    return (n + 2 * pad - effective_k) // stride + 1


def receptive_field(layers: list[tuple[int, int, int]]) -> tuple[int, int]:
    """Theoretical receptive field of a stack of (kernel, stride, dilation) layers.

    Recursion (``j`` = jump = product of strides so far, ``r`` = receptive field):
        r_{l} = r_{l-1} + (d_l (k_l − 1)) · j_{l-1},   j_l = j_{l-1} · s_l.
    Returns:
        (r, j) after the last layer, in input pixels.
    """
    r, j = 1, 1
    for k, s, d in layers:
        r = r + d * (k - 1) * j
        j = j * s
    return r, j


def conv_flops_and_params(
    c_in: int, c_out: int, k: int, h_out: int, w_out: int, groups: int = 1, bias: bool = True
) -> tuple[int, int]:
    """Multiply-adds and parameters of a ``k×k`` conv producing ``(c_out, h_out, w_out)``.

    MACs   = h_out · w_out · c_out · (c_in / groups) · k²
    params = c_out · (c_in / groups) · k²  (+ c_out for bias)
    Depthwise (groups = c_in = c_out) divides both by c_in.
    """
    per_output_macs = (c_in // groups) * k * k
    macs = h_out * w_out * c_out * per_output_macs
    params = c_out * per_output_macs + (c_out if bias else 0)
    return macs, params


def depthwise_separable_flops(c_in: int, c_out: int, k: int, h_out: int, w_out: int) -> tuple[int, int]:
    """MACs of ``k×k`` depthwise + ``1×1`` pointwise, and of the equivalent standard conv.

    Ratio ≈ 1/c_out + 1/k²  (MobileNet v1, eq. 5).
    """
    depthwise, _ = conv_flops_and_params(c_in, c_in, k, h_out, w_out, groups=c_in, bias=False)
    pointwise, _ = conv_flops_and_params(c_in, c_out, 1, h_out, w_out, bias=False)
    standard, _ = conv_flops_and_params(c_in, c_out, k, h_out, w_out, bias=False)
    return depthwise + pointwise, standard


# ---------------------------------------------------------------------------
# Naive forward — the definition, for reading
# ---------------------------------------------------------------------------


def conv2d_naive(x: np.ndarray, w: np.ndarray, stride: int = 1, pad: int = 0, dilation: int = 1) -> np.ndarray:
    """Cross-correlation by explicit loops (groups=1).  For understanding, not speed.

    ``out[b, o, i, j] = Σ_{c,u,v} x_pad[b, c, i·s + u·d, j·s + v·d] · w[o, c, u, v]``

    Args:
        x: (B, C_in, H, W).  w: (C_out, C_in, k_h, k_w).
    Returns:
        (B, C_out, H_out, W_out).
    """
    B, C_in, H, W = x.shape
    C_out, _, k_h, k_w = w.shape
    H_out = conv_output_size(H, k_h, stride, pad, dilation)
    W_out = conv_output_size(W, k_w, stride, pad, dilation)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # (B, C_in, H+2p, W+2p)
    out = np.zeros((B, C_out, H_out, W_out), dtype=x.dtype)  # (B, C_out, H_out, W_out)
    for b in range(B):
        for o in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    rows = i * stride + dilation * np.arange(k_h)  # (k_h,) input rows tapped
                    cols = j * stride + dilation * np.arange(k_w)  # (k_w,) input cols tapped
                    patch = x_pad[b][:, rows][:, :, cols]  # (C_in, k_h, k_w)
                    out[b, o, i, j] = np.sum(patch * w[o])
    return out


# ---------------------------------------------------------------------------
# im2col: convolution as one matrix multiply
# ---------------------------------------------------------------------------


def _tap_indices(H_out: int, W_out: int, k_h: int, k_w: int, stride: int, dilation: int):
    """Row/col indices of every (output position, kernel tap) pair.

    Returns:
        rows, cols: each (k_h·k_w, H_out·W_out) int arrays into the padded input.
    """
    u, v = np.meshgrid(np.arange(k_h), np.arange(k_w), indexing="ij")  # each (k_h, k_w)
    i, j = np.meshgrid(np.arange(H_out), np.arange(W_out), indexing="ij")  # each (H_out, W_out)
    rows = i.reshape(1, -1) * stride + u.reshape(-1, 1) * dilation  # (k_h·k_w, H_out·W_out)
    cols = j.reshape(1, -1) * stride + v.reshape(-1, 1) * dilation  # (k_h·k_w, H_out·W_out)
    return rows, cols


def im2col(x: np.ndarray, k_h: int, k_w: int, stride: int = 1, pad: int = 0, dilation: int = 1) -> np.ndarray:
    """Unfold every receptive field into a column (``torch.nn.Unfold``).

    Args:
        x: (B, C, H, W).
    Returns:
        (B, C·k_h·k_w, H_out·W_out) — column ``p`` holds the patch under output position ``p``,
        ordered ``(c, u, v)`` to match ``w.reshape(C_out, -1)``.
    """
    B, C, H, W = x.shape
    H_out = conv_output_size(H, k_h, stride, pad, dilation)
    W_out = conv_output_size(W, k_w, stride, pad, dilation)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))  # (B, C, H+2p, W+2p)
    rows, cols = _tap_indices(H_out, W_out, k_h, k_w, stride, dilation)  # (k_h·k_w, P) each
    cols_mat = x_pad[:, :, rows, cols]  # (B, C, k_h·k_w, P)  fancy-index gather
    return cols_mat.reshape(B, C * k_h * k_w, H_out * W_out)  # (B, C·k_h·k_w, P)


def col2im(
    cols: np.ndarray, x_shape: tuple[int, int, int, int], k_h: int, k_w: int, stride: int, pad: int, dilation: int
) -> np.ndarray:
    """Adjoint of :func:`im2col`: scatter-*add* columns back to image positions (``torch.nn.Fold``).

    Overlapping patches accumulate, which is exactly what the gradient w.r.t. the input needs.

    Args:
        cols: (B, C·k_h·k_w, H_out·W_out).
    Returns:
        (B, C, H, W) — padding rows/cols are cropped away.
    """
    B, C, H, W = x_shape
    H_out = conv_output_size(H, k_h, stride, pad, dilation)
    W_out = conv_output_size(W, k_w, stride, pad, dilation)
    rows, cols_idx = _tap_indices(H_out, W_out, k_h, k_w, stride, dilation)  # (k_h·k_w, P)
    x_pad = np.zeros((B, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)  # (B, C, H+2p, W+2p)
    cols4 = cols.reshape(B, C, k_h * k_w, H_out * W_out)  # (B, C, k_h·k_w, P)
    # np.add.at accumulates duplicates (plain fancy-index assignment would not)
    np.add.at(x_pad, (slice(None), slice(None), rows, cols_idx), cols4)
    return x_pad[:, :, pad : pad + H, pad : pad + W]  # (B, C, H, W)


def conv2d_im2col(
    x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None, stride: int = 1, pad: int = 0, dilation: int = 1, groups: int = 1
) -> tuple[np.ndarray, tuple]:
    """Forward pass as a GEMM per group: ``out_g = W_g @ cols_g``.

    Args:
        x: (B, C_in, H, W).  w: (C_out, C_in/groups, k_h, k_w).  b: (C_out,) or None.
    Returns:
        out: (B, C_out, H_out, W_out) and a cache for :func:`conv2d_backward`.
    """
    B, C_in, H, W = x.shape
    C_out, C_in_g, k_h, k_w = w.shape
    H_out = conv_output_size(H, k_h, stride, pad, dilation)
    W_out = conv_output_size(W, k_w, stride, pad, dilation)
    C_out_g = C_out // groups
    out = np.empty((B, C_out, H_out * W_out), dtype=x.dtype)  # (B, C_out, P)
    cols_per_group = []
    for g in range(groups):
        x_g = x[:, g * C_in_g : (g + 1) * C_in_g]  # (B, C_in/g, H, W)
        cols = im2col(x_g, k_h, k_w, stride, pad, dilation)  # (B, C_in/g·k_h·k_w, P)
        w_mat = w[g * C_out_g : (g + 1) * C_out_g].reshape(C_out_g, -1)  # (C_out/g, C_in/g·k_h·k_w)
        out[:, g * C_out_g : (g + 1) * C_out_g] = w_mat @ cols  # (B, C_out/g, P) batched GEMM
        cols_per_group.append(cols)
    if b is not None:
        out += b[None, :, None]  # broadcast (C_out,) over batch and positions
    out = out.reshape(B, C_out, H_out, W_out)  # (B, C_out, H_out, W_out)
    cache = (x.shape, w.shape, cols_per_group, stride, pad, dilation, groups)
    return out, cache


def conv2d_backward(dout: np.ndarray, w: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gradients of ``L`` w.r.t. input, weight and bias given ``dout = ∂L/∂out``.

    With ``out = W @ cols``:  ``∂L/∂W = dout @ colsᵀ``,  ``∂L/∂cols = Wᵀ @ dout``,
    then ``∂L/∂x = col2im(∂L/∂cols)`` (the scatter-add is the adjoint of the gather).

    Args:
        dout: (B, C_out, H_out, W_out).
    Returns:
        dx (B, C_in, H, W), dw (C_out, C_in/groups, k_h, k_w), db (C_out,).
    """
    x_shape, w_shape, cols_per_group, stride, pad, dilation, groups = cache
    B, C_in, H, W = x_shape
    C_out, C_in_g, k_h, k_w = w_shape
    C_out_g = C_out // groups
    dout2 = dout.reshape(B, C_out, -1)  # (B, C_out, P)
    dx = np.zeros(x_shape, dtype=dout.dtype)  # (B, C_in, H, W)
    dw = np.zeros(w_shape, dtype=dout.dtype)  # (C_out, C_in/g, k_h, k_w)
    for g in range(groups):
        dout_g = dout2[:, g * C_out_g : (g + 1) * C_out_g]  # (B, C_out/g, P)
        cols = cols_per_group[g]  # (B, C_in/g·k_h·k_w, P)
        w_mat = w[g * C_out_g : (g + 1) * C_out_g].reshape(C_out_g, -1)  # (C_out/g, C_in/g·k_h·k_w)
        dw_mat = np.einsum("bop,bkp->ok", dout_g, cols)  # (C_out/g, C_in/g·k_h·k_w): sum over batch b and positions p of dout[b,o,p]·cols[b,k,p]
        dw[g * C_out_g : (g + 1) * C_out_g] = dw_mat.reshape(C_out_g, C_in_g, k_h, k_w)
        dcols = w_mat.T @ dout_g  # (B, C_in/g·k_h·k_w, P)
        dx[:, g * C_in_g : (g + 1) * C_in_g] = col2im(
            dcols, (B, C_in_g, H, W), k_h, k_w, stride, pad, dilation
        )  # (B, C_in/g, H, W)
    db = dout.sum(axis=(0, 2, 3))  # (C_out,)
    return dx, dw, db


def conv_transpose2d(x: np.ndarray, w: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Transposed convolution = the input-gradient of a conv with the same weights.

    Args:
        x: (B, C_in, H, W) — plays the role of ``dout`` of the forward conv.
        w: (C_in, C_out, k, k) — PyTorch's ``ConvTranspose2d`` weight layout.
    Returns:
        (B, C_out, (H−1)·s − 2p + k, (W−1)·s − 2p + k).
    """
    B, C_in, H, W = x.shape
    _, C_out, k, _ = w.shape
    H_big = (H - 1) * stride - 2 * pad + k
    W_big = (W - 1) * stride - 2 * pad + k
    # a conv with weight (C_in, C_out, k, k) mapping (B, C_out, H_big, W_big) -> (B, C_in, H, W)
    w_fwd = w  # (C_in, C_out, k, k) already in "forward conv" layout for that direction
    dummy = np.zeros((B, C_out, H_big, W_big), dtype=x.dtype)  # (B, C_out, H_big, W_big)
    _, cache = conv2d_im2col(dummy, w_fwd, stride=stride, pad=pad)
    dx, _, _ = conv2d_backward(x, w_fwd, cache)  # (B, C_out, H_big, W_big)
    return dx
