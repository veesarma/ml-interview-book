"""Logit lens on the tiny residual LM trained on 'copy the token two back'."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mlbook.interp.logit_lens import TinyResidualLM, logit_lens, train_copy_task

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part15_logit_lens.png"


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    V, T = 16, 10
    model = TinyResidualLM(V, d=32, n_layers=4, max_len=T)
    train_copy_task(model, V, T, steps=400)
    tok = torch.randint(0, V, (1, T))
    lens = logit_lens(model, tok)  # (L+1, 1, T, V)
    probs = torch.softmax(lens[:, 0], dim=-1)  # (L+1, T, V)
    correct = torch.zeros(lens.shape[0], T)
    for t in range(2, T):
        correct[:, t] = probs[:, t, tok[0, t - 2]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(correct[:, 2:], cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(lens.shape[0])); ax.set_yticklabels(["embed"] + [f"after layer {l}" for l in range(1, lens.shape[0])])
    ax.set_xticks(range(T - 2)); ax.set_xticklabels([f"pos {t}" for t in range(2, T)], fontsize=8)
    for l in range(correct.shape[0]):
        for t in range(T - 2):
            ax.text(t, l, f"{correct[l, t + 2]:.2f}", ha="center", va="center", fontsize=7, color="white" if correct[l, t + 2] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="P(correct token) read by the lens")
    ax.set_title("Logit lens: the answer becomes decodable from the residual stream layer by layer", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
