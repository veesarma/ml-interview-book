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


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, float, dict]:
    """Top-label ECE and MCE with equal-width bins.

    probs (N, K) softmax outputs, labels (N,). Returns (ece, mce, bins) where bins
    holds per-bin confidence, accuracy and count for a reliability diagram.
    """
    conf = probs.max(axis=1)  # (N,)
    pred = probs.argmax(axis=1)  # (N,)
    correct = (pred == labels).astype(float)  # (N,)
    edges = np.linspace(0.0, 1.0, n_bins + 1)  # (B+1,)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)  # (N,)
    ece, mce = 0.0, 0.0
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    N = len(conf)
    for b in range(n_bins):
        m = idx == b
        bin_count[b] = m.sum()
        if bin_count[b] == 0:
            continue
        bin_conf[b] = conf[m].mean()
        bin_acc[b] = correct[m].mean()
        gap = abs(bin_acc[b] - bin_conf[b])
        ece += bin_count[b] / N * gap
        mce = max(mce, gap)
    return float(ece), float(mce), {"conf": bin_conf, "acc": bin_acc, "count": bin_count, "edges": edges}


def adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE with equal-mass bins (each bin holds N / n_bins examples), which avoids
    the empty-high-confidence-bin pathology of equal-width binning."""
    conf = probs.max(axis=1)  # (N,)
    correct = (probs.argmax(axis=1) == labels).astype(float)  # (N,)
    order = np.argsort(conf)
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        ece += chunk.size / len(conf) * abs(correct[chunk].mean() - conf[chunk].mean())
    return float(ece)


class TemperatureScaling(nn.Module):
    """Divide logits by a learned scalar temperature. (N, K) -> (N, K)."""

    def __init__(self) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # (1,) log T, T = 1 at init

    @property
    def temperature(self) -> float:
        return float(self.log_t.detach().exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()  # (N, K)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> "TemperatureScaling":
        """Minimise validation NLL over T with L-BFGS. logits (N, K), labels (N,)."""
        logits = logits.detach()
        opt = torch.optim.LBFGS([self.log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self
