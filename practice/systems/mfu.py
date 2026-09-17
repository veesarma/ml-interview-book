# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/systems/mfu.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k mfu -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py systems/mfu --force

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
PEAK_BF16_DENSE_FLOPS = {'A100': 312000000000000.0, 'H100_SXM': 989000000000000.0}

def mfu(tokens_per_second: float, cfg: TransformerConfig, seq_len: int, peak_flops: float, n_gpus: int=1) -> float:
    """MFU in [0, 1] from observed throughput on ``n_gpus`` GPUs."""
    raise NotImplementedError('TODO: implement mfu (see the reference in src/mlbook)')

def hfu(tokens_per_second: float, cfg: TransformerConfig, seq_len: int, peak_flops: float, n_gpus: int=1, recompute: bool=True) -> float:
    """HFU: like MFU but counts the recomputed forward when checkpointing is on."""
    raise NotImplementedError('TODO: implement hfu (see the reference in src/mlbook)')

def mfu_from_six_nd(n_params: float, tokens_per_second: float, peak_flops: float, n_gpus: int=1) -> float:
    """Back-of-envelope MFU using C = 6 N D (no attention term)."""
    raise NotImplementedError('TODO: implement mfu_from_six_nd (see the reference in src/mlbook)')

def training_days(n_params: float, tokens: float, peak_flops: float, n_gpus: int, target_mfu: float) -> float:
    """Wall-clock days for 6 N D FLOPs at ``target_mfu`` on ``n_gpus`` GPUs."""
    raise NotImplementedError('TODO: implement training_days (see the reference in src/mlbook)')
