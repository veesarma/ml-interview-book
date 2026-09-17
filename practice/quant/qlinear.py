# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/quant/qlinear.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k qlinear -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py quant/qlinear --force

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
    raise NotImplementedError('TODO: implement pack_int4 (see the reference in src/mlbook)')

def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of ``pack_int4``: (out, in // 2) uint8 -> (out, in) int32 in [-8, 7]."""
    raise NotImplementedError('TODO: implement unpack_int4 (see the reference in src/mlbook)')

class QuantizedLinear(nn.Module):
    """Weight-only INT8/INT4 Linear with per-group symmetric scales; y = x Ŵ^T + b.

    Weights are stored as integer codes (INT4 packed two per byte) and a scale per
    group; the forward pass dequantises to the activation dtype and does a normal
    matmul.  This is exactly what GPTQ/AWQ checkpoints look like on disk.
    """

    def __init__(self, in_features: int, out_features: int, bits: int, group_size: int, bias: bool=True) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @classmethod
    def from_linear(cls, lin: nn.Linear, bits: int, group_size: int) -> 'QuantizedLinear':
        """Round-to-nearest quantise an existing nn.Linear."""
        raise NotImplementedError('TODO: implement from_linear (see the reference in src/mlbook)')

    def dequantized_weight(self) -> torch.Tensor:
        """Ŵ of shape (out, in) in float32."""
        raise NotImplementedError('TODO: implement dequantized_weight (see the reference in src/mlbook)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(…, in) -> (…, out)."""
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def weight_bytes(self) -> int:
        """Bytes of codes + scales (fp32 scales here; production stores fp16 scales)."""
        raise NotImplementedError('TODO: implement weight_bytes (see the reference in src/mlbook)')

def int8_matmul_with_outliers(X: torch.Tensor, W: torch.Tensor, threshold: float=6.0) -> torch.Tensor:
    """LLM.int8() mixed decomposition of X W^T.

    Input columns whose absolute activation exceeds ``threshold`` anywhere in the
    batch are computed in fp16/fp32; every other column goes through row-wise
    (per token) INT8 activations × column-wise (per output) INT8 weights.

    Args:
        X: (N, in) activations, W: (out, in) weights.
    Returns:
        (N, out).
    """
    raise NotImplementedError('TODO: implement int8_matmul_with_outliers (see the reference in src/mlbook)')

def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group_size: int, damp: float=0.01) -> torch.Tensor:
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
    raise NotImplementedError('TODO: implement gptq_quantize (see the reference in src/mlbook)')
