"""Attention weights of a seq2seq model trained to reverse a sequence: the learned alignment is
the anti-diagonal. Writes docs/assets/figures/part05_attention_heatmap.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from mlbook.sequence.seq2seq_attention import Seq2SeqAttention  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_attention_heatmap.png"
PAD, BOS, EOS, V, T = 0, 1, 2, 10, 7


def batch(B: int, g: torch.Generator):
    src = torch.randint(3, V, (B, T), generator=g)
    tgt = torch.flip(src, dims=[1])
    return src, torch.cat([torch.full((B, 1), BOS), tgt], 1), torch.cat([tgt, torch.full((B, 1), EOS)], 1)


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(0)
    model = Seq2SeqAttention(V, d_model=32, attention="additive")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        src, tgt_in, tgt_out = batch(64, g)
        logits, _ = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    src, tgt_in, tgt_out = batch(1, g)
    with torch.no_grad():
        _, attn = model(src, tgt_in)  # (1, T+1, T)
    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor="white")
    im = ax.imshow(attn[0].numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(T))
    ax.set_xticklabels([str(int(s)) for s in src[0]])
    ax.set_yticks(range(T + 1))
    ax.set_yticklabels([str(int(s)) if s != EOS else "<eos>" for s in tgt_out[0]])
    ax.set_xlabel("source tokens (keys)")
    ax.set_ylabel("generated tokens (queries)")
    ax.set_title("Additive attention learned on 'reverse the sequence'", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\alpha_{t,j}$")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
