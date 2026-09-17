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
        return self.peak_flops / self.bandwidth

    def attainable(self, intensity: float) -> float:
        return min(self.peak_flops, intensity * self.bandwidth)


H100_SXM = Machine("H100 SXM (BF16 dense, HBM3)", 989e12, 3.35e12)
A100_80GB = Machine("A100 80GB (BF16 dense, HBM2e)", 312e12, 2.0e12)


@dataclass(frozen=True)
class Kernel:
    name: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes


def matmul_kernel(M: int, N: int, K: int, bytes_per_elem: float = 2.0, name: str | None = None) -> Kernel:
    """(M, K) @ (K, N): 2MNK FLOPs; reads A and B once, writes C once."""
    flops = 2.0 * M * N * K
    moved = bytes_per_elem * (M * K + K * N + M * N)
    return Kernel(name or f"matmul {M}x{K}x{N}", flops, moved)


def decode_step_kernel(n_params: float, batch: int, bytes_per_param: float = 2.0) -> Kernel:
    """One decode step: every weight read once, 2 FLOPs per weight per sequence."""
    return Kernel(f"decode B={batch}", 2.0 * n_params * batch, bytes_per_param * n_params)


def layernorm_kernel(n_tokens: int, d: int, bytes_per_elem: float = 2.0) -> Kernel:
    """~ 8 FLOPs per element (mean, var, normalise, scale/shift); read x, write y."""
    n = n_tokens * d
    return Kernel(f"layernorm {n_tokens}x{d}", 8.0 * n, 2.0 * bytes_per_elem * n)


def naive_attention_kernel(T: int, d: int, bytes_per_elem: float = 2.0) -> Kernel:
    """QK^T + softmax + PV with the (T, T) matrix materialised in HBM (single head)."""
    flops = 2.0 * T * T * d * 2 + 5.0 * T * T
    moved = bytes_per_elem * (3 * T * d + 4 * T * T + T * d)  # Q,K,V in; S out+in, P out+in; O out
    return Kernel(f"naive attention T={T}", flops, moved)


def flash_attention_kernel(T: int, d: int, bytes_per_elem: float = 2.0) -> Kernel:
    """Same FLOPs, but the (T, T) tensor never leaves SRAM: only Q, K, V, O touch HBM."""
    flops = 2.0 * T * T * d * 2 + 5.0 * T * T
    moved = bytes_per_elem * (4 * T * d)
    return Kernel(f"flash attention T={T}", flops, moved)


def time_lower_bound(kernel: Kernel, m: Machine) -> float:
    """Roofline time: max(compute time, memory time)."""
    return max(kernel.flops / m.peak_flops, kernel.bytes / m.bandwidth)


def plot_roofline(m: Machine, kernels: list[Kernel], path: str) -> None:
    """Write a log-log roofline with each kernel placed at (intensity, attainable)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    I = np.logspace(-1, 4, 400)  # (400,)
    roof = np.minimum(m.peak_flops, I * m.bandwidth) / 1e12  # (400,) TFLOP/s
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.loglog(I, roof, color="k", lw=2, label=m.name)
    ax.axvline(m.ridge_point, color="0.6", ls="--", lw=1)
    ax.text(m.ridge_point * 1.1, roof.min() * 1.5, f"ridge = {m.ridge_point:.0f} FLOP/B", fontsize=9, color="0.4")
    for i, k in enumerate(kernels):
        x = k.intensity
        y = m.attainable(x) / 1e12
        ax.plot(x, y, "o", color=f"C{i % 10}", ms=8)
        ax.annotate(k.name, (x, y), textcoords="offset points", xytext=(6, -12 if i % 2 else 6), fontsize=8)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of HBM traffic)")
    ax.set_ylabel("attainable TFLOP/s")
    ax.set_title("Roofline: memory-bound left of the ridge, compute-bound right")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
