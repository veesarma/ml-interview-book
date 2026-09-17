"""Gradient norm per layer at initialisation for a 12-block Pre-LN vs Post-LN encoder stack.
Post-LN puts large gradients near the output (hence warm-up), Pre-LN is flat.
Writes docs/assets/figures/part05_pre_post_ln.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mlbook.transformer.blocks import TransformerEncoderBlock  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_pre_post_ln.png"


def grad_norms(pre_norm: bool, L: int = 12, d: int = 64, n_trials: int = 8) -> np.ndarray:
    out = np.zeros((n_trials, L))
    for trial in range(n_trials):
        torch.manual_seed(trial)
        blocks = torch.nn.ModuleList([TransformerEncoderBlock(d, 4, pre_norm=pre_norm) for _ in range(L)])
        head = torch.nn.Linear(d, 10)
        x = torch.randn(8, 16, d)
        h = x
        for b in blocks:
            h = b(h)
        if pre_norm:
            h = torch.nn.functional.layer_norm(h, (d,))  # final LN, as in GPT-2 / LLaMA
        loss = torch.nn.functional.cross_entropy(head(h).reshape(-1, 10), torch.randint(0, 10, (8 * 16,)))
        loss.backward()
        for i, b in enumerate(blocks):
            out[trial, i] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in b.ffn.parameters())))
    return out


def main() -> None:
    torch.set_num_threads(1)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    layers = np.arange(1, 13)
    for pre, label in ((False, "Post-LN (original Transformer, BERT)"), (True, "Pre-LN (GPT-2, LLaMA)")):
        g = grad_norms(pre)
        ax.semilogy(layers, g.mean(0), marker="o", label=label)
        ax.fill_between(layers, g.min(0), g.max(0), alpha=0.2)
    ax.set_xlabel("block index (1 = closest to the input)")
    ax.set_ylabel("||grad of FFN parameters|| at init")
    ax.set_title("Gradient magnitude by depth at initialisation (12 blocks, d=64, 8 seeds)", loc="left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
