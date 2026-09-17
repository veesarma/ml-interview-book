"""Weight-only quantised Linear layers that dequantise on the fly, INT4 nibble packing,
LLM.int8()-style mixed-precision decomposition, and a compact GPTQ.

At inference the bottleneck of decode is reading weights from HBM, so storing W in
4 bits and expanding it inside the kernel (here: in Python, for clarity) cuts memory
traffic by 4× versus FP16 while the arithmetic stays in FP16/BF16.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mlbook.quant.quantize import dequantize_per_group, quantize_per_group


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit codes in [-8, 7] two per byte: low nibble = even column, high = odd.

    Args:
        q: (out, in) int32 with even ``in``.
    Returns:
        (out, in // 2) uint8.
    """
    u = (q + 8).to(torch.uint8)  # (out, in) unsigned nibbles in [0, 15]
    return u[:, 0::2] | (u[:, 1::2] << 4)  # (out, in // 2)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_int4``: (out, in // 2) uint8 -> (out, in) int32 in [-8, 7]."""
    low = (packed & 0x0F).to(torch.int32) - 8  # (out, in // 2)
    high = (packed >> 4).to(torch.int32) - 8  # (out, in // 2)
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)  # (out, in)


class QuantizedLinear(nn.Module):
    """Weight-only INT8/INT4 Linear with per-group symmetric scales; y = x Ŵ^T + b.

    Weights are stored as integer codes (INT4 packed two per byte) and a scale per
    group; the forward pass dequantises to the activation dtype and does a normal
    matmul.  This is exactly what GPTQ/AWQ checkpoints look like on disk.
    """

    def __init__(self, in_features: int, out_features: int, bits: int, group_size: int, bias: bool = True) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        self.in_features, self.out_features = in_features, out_features
        self.bits, self.group_size = bits, group_size
        n_groups = in_features // group_size
        cols = in_features // 2 if bits == 4 else in_features
        code_dtype = torch.uint8 if bits == 4 else torch.int8
        self.register_buffer("codes", torch.zeros(out_features, cols, dtype=code_dtype))  # (out, in or in/2)
        self.register_buffer("scale", torch.ones(out_features, n_groups, 1))  # (out, n_groups, 1)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None  # (out,)

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int, group_size: int) -> "QuantizedLinear":
        """Round-to-nearest quantise an existing nn.Linear."""
        qlin = cls(lin.in_features, lin.out_features, bits, group_size, bias=lin.bias is not None)
        r = quantize_per_group(lin.weight.detach(), bits, group_size, symmetric=True)
        qlin.codes.copy_(pack_int4(r.q) if bits == 4 else r.q.to(torch.int8))
        qlin.scale.copy_(r.scale)
        if lin.bias is not None:
            qlin.bias.data.copy_(lin.bias.detach())
        return qlin

    def dequantized_weight(self) -> torch.Tensor:
        """Ŵ of shape (out, in) in float32."""
        q = unpack_int4(self.codes) if self.bits == 4 else self.codes.to(torch.int32)  # (out, in)
        grouped = q.reshape(self.out_features, -1, self.group_size).float()  # (out, n_groups, group)
        return (grouped * self.scale).reshape(self.out_features, self.in_features)  # (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, out)."""
        W_hat = self.dequantized_weight()  # (out, in) rebuilt on the fly
        y = x @ W_hat.T  # (…, out)
        return y if self.bias is None else y + self.bias

    def weight_bytes(self) -> int:
        """Bytes of codes + scales (fp32 scales here; production stores fp16 scales)."""
        return self.codes.numel() * self.codes.element_size() + self.scale.numel() * 4


def int8_matmul_with_outliers(X: torch.Tensor, W: torch.Tensor, threshold: float = 6.0) -> torch.Tensor:
    """LLM.int8() mixed decomposition of X W^T.

    Input columns whose absolute activation exceeds ``threshold`` anywhere in the
    batch are computed in fp16/fp32; every other column goes through row-wise
    (per token) INT8 activations × column-wise (per output) INT8 weights.

    Args:
        X: (N, in) activations, W: (out, in) weights.
    Returns:
        (N, out).
    """
    outlier = X.abs().amax(dim=0) > threshold  # (in,) which input features are outliers
    X_o, W_o = X[:, outlier], W[:, outlier]  # (N, n_o), (out, n_o)
    X_r, W_r = X[:, ~outlier], W[:, ~outlier]  # (N, in-n_o), (out, in-n_o)
    sx = X_r.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127  # (N, 1) per-row scale
    sw = W_r.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127  # (out, 1) per-column scale
    Xq = torch.round(X_r / sx)  # (N, in-n_o) integers in [-127, 127]
    Wq = torch.round(W_r / sw)  # (out, in-n_o)
    regular = (Xq @ Wq.T) * sx * sw.T  # (N, out) int accumulate, then rescale
    return regular + X_o @ W_o.T  # (N, out)


def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group_size: int, damp: float = 0.01) -> torch.Tensor:
    """Compact GPTQ (OBQ with a fixed column order): quantise W column by column,
    feeding each column's rounding error into the not-yet-quantised columns
    through the inverse Hessian ``H^{-1}`` of the layer's input second moments.

    For column j:   q_j = round(w_j / s),  err_j = (w_j − ŵ_j) / [H^{-1}]_{jj},
                    W[:, j+1:] −= err_j ⊗ [H^{-1}]_{j, j+1:}.

    Args:
        W: (out, in) weight; H: (in, in) = X^T X / N from calibration activations.
    Returns:
        (out, in) dequantised weight (codes ⋅ scales) with symmetric per-group scales.
    """
    W = W.clone().float()
    H = H.clone().float()
    H += damp * H.diagonal().mean() * torch.eye(H.shape[0])  # dampening for invertibility
    H_inv = torch.linalg.inv(H)  # (in, in)
    q_max = 2 ** (bits - 1) - 1
    out_f, in_f = W.shape
    W_hat = torch.zeros_like(W)  # (out, in)
    scale = torch.zeros(out_f, 1)  # (out, 1) current group's scale
    for j in range(in_f):
        if j % group_size == 0:  # scales come from the *updated* weights of this group
            scale = W[:, j : j + group_size].abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / q_max
        w = W[:, j : j + 1]  # (out, 1)
        w_hat = torch.clamp(torch.round(w / scale), -q_max - 1, q_max) * scale  # (out, 1)
        W_hat[:, j : j + 1] = w_hat
        err = (w - w_hat) / H_inv[j, j]  # (out, 1)
        W[:, j + 1 :] -= err @ H_inv[j : j + 1, j + 1 :]  # (out, in-j-1) push error forward
    return W_hat
