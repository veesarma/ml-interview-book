"""MAE: what 75% masking looks like, and how reconstruction quality moves with the mask ratio.

Writes docs/assets/figures/part10_mae_masking.png.  Run:  python figures/part10_mae_masking.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mlbook.ssl import mae as M  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part10_mae_masking.png"
torch.set_num_threads(1)
SIZE, PATCH = 32, 4
RATIOS = (0.25, 0.5, 0.75, 0.9)


def stripe_image() -> torch.Tensor:
    """A synthetic image with structure a small model can learn: diagonal stripes plus a blob."""
    yy, xx = torch.meshgrid(torch.linspace(0, 1, SIZE), torch.linspace(0, 1, SIZE), indexing="ij")
    img = 0.5 + 0.5 * torch.sin(12 * (xx + yy))                           # (H, W)
    img = img + torch.exp(-40 * ((xx - 0.3) ** 2 + (yy - 0.7) ** 2))      # (H, W)
    return img[None, None]                                                 # (1, 1, H, W)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    torch.manual_seed(0)
    img = stripe_image()

    fig = plt.figure(figsize=(14, 5.6), facecolor="white")
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.95], hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img[0, 0], cmap="gray"); ax.set_title("input", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    for i, r in enumerate(RATIOS):
        ax = fig.add_subplot(gs[0, i + 1])
        tokens = M.patchify(img, PATCH)                                    # (1, N, P*P*C)
        _, mask, _ = M.random_masking(tokens, r)                           # (1, N)
        shown = tokens.clone()
        shown[0][mask[0] > 0.5] = 0.5                                      # grey out masked patches
        ax.imshow(M.unpatchify(shown, PATCH, 1, SIZE, SIZE)[0, 0], cmap="gray", vmin=0, vmax=1.6)
        n_vis = int((mask[0] == 0).sum())
        ax.set_title(f"mask {int(r * 100)}%\nencoder sees {n_vis} of {tokens.shape[1]} patches", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # train a tiny MAE at each ratio on a small set of these images and report held-out masked MSE
    ax = fig.add_subplot(gs[1, :2])
    imgs = torch.cat([stripe_image() * s for s in torch.linspace(0.7, 1.3, 12)])   # (12, 1, H, W)
    final = []
    for r in RATIOS:
        torch.manual_seed(0)
        model = M.MAE(img_size=SIZE, patch=PATCH, in_chans=1, d_enc=64, d_dec=48, mask_ratio=r)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        for _ in range(300):
            loss, _, _ = model(imgs)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            final.append(float(torch.stack([model(imgs)[0] for _ in range(5)]).mean()))
    ax.bar([f"{int(r * 100)}%" for r in RATIOS], final, color=[colors[i] for i in range(4)], alpha=0.85)
    ax.set_xlabel("mask ratio"); ax.set_ylabel("masked-patch MSE after 300 steps")
    ax.set_title("On a globally predictable image the ratio barely moves the loss.\nReconstruction error is not a measure of feature quality.", fontsize=9.5)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    ax.text(0.0, 1.0,
            "Why the asymmetry pays\n\n"
            "Encoder cost scales with the number of tokens it attends over.\n"
            "At a 75% mask ratio the encoder runs on 25% of the patches, and\n"
            "attention is quadratic in token count, so its attention cost drops\n"
            "by roughly 16x. The decoder is narrow and shallow (a few blocks\n"
            "against the encoder's dozens) and is thrown away after pretraining.\n\n"
            "The mask ratio is a task-difficulty knob. Too low and the model\n"
            "interpolates neighbouring pixels, which teaches it nothing about\n"
            "objects. Too high and there is not enough context left to predict\n"
            "from. He et al. report 75% as the sweet spot for ImageNet.",
            va="top", ha="left", fontsize=8.5, color="#333333", transform=ax.transAxes)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
