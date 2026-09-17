# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/reliability/calibration.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k calibration -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py reliability/calibration --force

"""Calibration: ECE / MCE, reliability-diagram bins, and temperature scaling.

ECE = sum_b (|B_b| / N) | acc(B_b) - conf(B_b) |    over confidence bins B_b.
Temperature scaling (Guo et al. 2017) fits one scalar T > 0 minimising the NLL
of softmax(z / T) on a held-out validation set. It changes confidence, never the
argmax, so accuracy is unchanged.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int=15) -> tuple[float, float, dict]:
    """Top-label ECE and MCE with equal-width bins.

    probs (N, K) softmax outputs, labels (N,). Returns (ece, mce, bins) where bins
    holds per-bin confidence, accuracy and count for a reliability diagram.
    """
    raise NotImplementedError('TODO: implement expected_calibration_error (see the reference in src/mlbook)')

def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int=15) -> float:
    """ECE with equal-mass bins (each bin holds N / n_bins examples), which avoids
    the empty-high-confidence-bin pathology of equal-width binning."""
    raise NotImplementedError('TODO: implement adaptive_ece (see the reference in src/mlbook)')

class TemperatureScaling(nn.Module):
    """Divide logits by a learned scalar temperature. (N, K) -> (N, K)."""

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    @property
    def temperature(self) -> float:
        raise NotImplementedError('TODO: implement temperature (see the reference in src/mlbook)')

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('TODO: implement forward (see the reference in src/mlbook)')

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int=100) -> 'TemperatureScaling':
        """Minimise validation NLL over T with L-BFGS. logits (N, K), labels (N,)."""
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')
