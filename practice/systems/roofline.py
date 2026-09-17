# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/roofline.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k roofline -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/roofline --force

"""The roofline model.

    attainable FLOP/s = min(peak_flops, intensity * bandwidth)
    intensity I = FLOPs / bytes moved (FLOP per byte)
    ridge point I* = peak_flops / bandwidth: kernels left of it are memory-bound.

Kernel intensities (16-bit, 2 bytes per element):
    matmul (M, K) x (K, N): 2 M N K / (2 (M K + K N + M N))
    decode step of an LLM (batch B): 2 P B / (2 P) = B
    elementwise / layernorm on n elements: O(1) FLOP per byte (e.g. ~ 1/2 ... 2)
    naive attention scores (T x T x d): 2 T^2 d / (2 (2 T d + T^2)) ~ d for T >> d ... but
    with the (T, T) probabilities written and re-read the intensity collapses toward ~1.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Machine:
    name: str
    peak_flops: float
    bandwidth: float

    @property
    def ridge_point(self) -> float:
        raise NotImplementedError('TODO: implement ridge_point (see the reference in src/mlbook)')

    def attainable(self, intensity: float) -> float:
        raise NotImplementedError('TODO: implement attainable (see the reference in src/mlbook)')
H100_SXM = Machine('H100 SXM (BF16 dense, HBM3)', 989000000000000.0, 3350000000000.0)
A100_80GB = Machine('A100 80GB (BF16 dense, HBM2e)', 312000000000000.0, 2000000000000.0)

@dataclass(frozen=True)
class Kernel:
    name: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        raise NotImplementedError('TODO: implement intensity (see the reference in src/mlbook)')

def matmul_kernel(M: int, N: int, K: int, bytes_per_elem: float=2.0, name: str | None=None) -> Kernel:
    """(M, K) @ (K, N): 2MNK FLOPs; reads A and B once, writes C once."""
    raise NotImplementedError('TODO: implement matmul_kernel (see the reference in src/mlbook)')

def decode_step_kernel(n_params: float, batch: int, bytes_per_param: float=2.0) -> Kernel:
    """One decode step: every weight read once, 2 FLOPs per weight per sequence."""
    raise NotImplementedError('TODO: implement decode_step_kernel (see the reference in src/mlbook)')

def layernorm_kernel(n_tokens: int, d: int, bytes_per_elem: float=2.0) -> Kernel:
    """~ 8 FLOPs per element (mean, var, normalise, scale/shift); read x, write y."""
    raise NotImplementedError('TODO: implement layernorm_kernel (see the reference in src/mlbook)')

def naive_attention_kernel(T: int, d: int, bytes_per_elem: float=2.0) -> Kernel:
    """QK^T + softmax + PV with the (T, T) matrix materialised in HBM (single head)."""
    raise NotImplementedError('TODO: implement naive_attention_kernel (see the reference in src/mlbook)')

def flash_attention_kernel(T: int, d: int, bytes_per_elem: float=2.0) -> Kernel:
    """Same FLOPs, but the (T, T) tensor never leaves SRAM: only Q, K, V, O touch HBM."""
    raise NotImplementedError('TODO: implement flash_attention_kernel (see the reference in src/mlbook)')

def time_lower_bound(kernel: Kernel, m: Machine) -> float:
    """Roofline time: max(compute time, memory time)."""
    raise NotImplementedError('TODO: implement time_lower_bound (see the reference in src/mlbook)')

def plot_roofline(m: Machine, kernels: list[Kernel], path: str) -> None:
    """Write a log-log roofline with each kernel placed at (intensity, attainable)."""
    raise NotImplementedError('TODO: implement plot_roofline (see the reference in src/mlbook)')
