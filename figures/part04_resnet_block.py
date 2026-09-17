"""Residual block diagram and the loss-surface intuition: plain vs residual depth scaling."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

OUT = "docs/assets/figures/part04_resnet_block.png"


def grad_norm_vs_depth(residual: bool, depth: int, width: int = 64, trials: int = 5):
    """Norm of ∂output/∂input for a stack of tanh layers with/without identity skips."""
    norms = []
    for t in range(trials):
        torch.manual_seed(t)
        layers = [nn.Linear(width, width) for _ in range(depth)]
        x = torch.randn(1, width, requires_grad=True)
        h = x
        for lin in layers:
            f = torch.tanh(lin(h))
            h = h + f if residual else f
        g = torch.autograd.grad(h.sum(), x)[0]
        norms.append(g.norm().item())
    return float(np.mean(norms))


def main():
    torch.set_num_threads(1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    ax1.set_xlim(0, 13)
    ax1.set_ylim(0, 6)
    ax1.axis("off")
    boxes = [("x_l", 0.7, 0.9), ("conv 3×3\nBN, ReLU", 3.2, 2.2), ("conv 3×3\nBN", 6.4, 2.2), ("+", 9.0, 0.7), ("ReLU\n→ x_{l+1}", 11.4, 1.8)]
    for label, xc, w in boxes:
        ax1.add_patch(patches.FancyBboxPatch((xc - w / 2, 2.4), w, 1.2, boxstyle="round,pad=0.05", fc="#e8eefc", ec="#1f2937"))
        ax1.text(xc, 3.0, label, ha="center", va="center", fontsize=9)
    for (a, b) in [(1.15, 2.1), (4.3, 5.3), (7.5, 8.65), (9.35, 10.5)]:
        ax1.annotate("", xy=(b, 3.0), xytext=(a, 3.0), arrowprops=dict(arrowstyle="->"))
    ax1.annotate("", xy=(9.0, 3.65), xytext=(0.7, 3.65), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color="#059669", lw=2))
    ax1.text(4.85, 5.5, "identity shortcut: ∂x_{l+1}/∂x_l = I + ∂F/∂x_l", color="#059669", ha="center", fontsize=10)
    ax1.text(4.85, 1.7, "residual branch F(x_l)", ha="center", fontsize=10)
    ax1.set_title("Basic residual block")
    depths = [2, 4, 8, 16, 32, 64]
    ax2.semilogy(depths, [grad_norm_vs_depth(False, d) for d in depths], marker="o", label="plain: h ← tanh(W h)")
    ax2.semilogy(depths, [grad_norm_vs_depth(True, d) for d in depths], marker="s", label="residual: h ← h + tanh(W h)")
    ax2.set_xlabel("depth (layers)")
    ax2.set_ylabel("‖∂output/∂input‖ at init")
    ax2.set_title("Identity path keeps the input-gradient alive at depth")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
