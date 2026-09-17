"""Grad-CAM on a tiny CNN trained to detect a bright square in a noisy 16x16 image."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from mlbook.interp.gradcam import TinyCNN, grad_cam
from mlbook.interp.saliency import saliency_map, smoothgrad, vanilla_gradient

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part15_gradcam.png"


def make_data(n: int, g: torch.Generator):
    X = 0.3 * torch.rand(n, 1, 16, 16, generator=g)  # (n, 1, 16, 16) noise
    y = torch.randint(0, 2, (n,), generator=g)
    for i in range(n):
        if y[i] == 1:
            r, c = torch.randint(0, 12, (2,), generator=g)
            X[i, 0, r:r + 4, c:c + 4] = 1.0
    return X, y


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(0)
    X, y = make_data(512, g)
    model = TinyCNN(2, width=8)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(150):
        loss = F.cross_entropy(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    x = torch.zeros(1, 16, 16) + 0.3 * torch.rand(1, 16, 16, generator=g)
    x[0, 3:7, 9:13] = 1.0
    cam = grad_cam(model, x, target=1)
    sal = saliency_map(vanilla_gradient(model, x, 1))
    sg = saliency_map(smoothgrad(model, x, 1, n_samples=32, sigma=0.2))
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    for ax, img, title in zip(axes, [x[0], sal, sg, cam], ["input (class 1: square present)", "vanilla gradient |grad|", "SmoothGrad", "Grad-CAM (conv2, upsampled)"]):
        ax.imshow(img.detach(), cmap="gray" if title.startswith("input") else "magma")
        if not title.startswith("input"):
            ax.imshow(x[0], cmap="gray", alpha=0.25)
        ax.set_title(title, fontsize=9); ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
