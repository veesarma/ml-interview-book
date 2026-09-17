"""Model FLOPs utilisation (MFU) and hardware FLOPs utilisation (HFU).

    MFU = (tokens/s * model_flops_per_token) / peak_flops_per_s
    HFU = (tokens/s * hardware_flops_per_token) / peak_flops_per_s

``model_flops_per_token`` = 6 N + 12 L h s counts only the FLOPs the maths requires;
``hardware_flops_per_token`` also counts recomputation (8 N + 16 L h s with full
activation checkpointing). MFU is the number to quote (PaLM, Chowdhery et al. 2022):
it does not reward doing extra work.
"""

from __future__ import annotations

from mlbook.systems.flops import flops_per_token
from mlbook.systems.memory_calc import TransformerConfig

PEAK_BF16_DENSE_FLOPS = {
    # Dense (non-sparse) BF16 tensor-core peak per GPU, from NVIDIA datasheets.
    "A100": 312e12,
    "H100_SXM": 989e12,
}


def mfu(tokens_per_second: float, cfg: TransformerConfig, seq_len: int, peak_flops: float, n_gpus: int = 1) -> float:
    """MFU in [0, 1] from observed throughput on ``n_gpus`` GPUs."""
    achieved = tokens_per_second * flops_per_token(cfg, seq_len, "train", recompute=False)
    return achieved / (peak_flops * n_gpus)


def hfu(tokens_per_second: float, cfg: TransformerConfig, seq_len: int, peak_flops: float, n_gpus: int = 1, recompute: bool = True) -> float:
    """HFU: like MFU but counts the recomputed forward when checkpointing is on."""
    achieved = tokens_per_second * flops_per_token(cfg, seq_len, "train", recompute=recompute)
    return achieved / (peak_flops * n_gpus)


def mfu_from_six_nd(n_params: float, tokens_per_second: float, peak_flops: float, n_gpus: int = 1) -> float:
    """Back-of-envelope MFU using C = 6 N D (no attention term)."""
    return 6.0 * n_params * tokens_per_second / (peak_flops * n_gpus)


def training_days(n_params: float, tokens: float, peak_flops: float, n_gpus: int, target_mfu: float) -> float:
    """Wall-clock days for 6 N D FLOPs at ``target_mfu`` on ``n_gpus`` GPUs."""
    seconds = 6.0 * n_params * tokens / (peak_flops * n_gpus * target_mfu)
    return seconds / 86400.0
