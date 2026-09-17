"""FGSM on a tiny 2-D classifier: decision boundary, clean points, and the
eps-sized perturbation that flips them. Writes docs/assets/figures/part13_fgsm.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from mlbook.reliability.adversarial import TinyMLP, fgsm, pgd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part13_fgsm.png"


def main() -> None:
    torch.manual_seed(0)
    x = torch.rand(400, 2)
    y = ((x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 < 0.09).long()  # disc vs outside
    model = TinyMLP(2, 64, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    eps = 0.06
    x_f = fgsm(model, x, y, eps)
    x_p = pgd(model, x, y, eps, alpha=0.015, n_steps=20)
    with torch.no_grad():
        acc = lambda xx: (model(xx).argmax(1) == y).float().mean().item()  # noqa: E731
        accs = (acc(x), acc(x_f), acc(x_p))
        gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
        grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
        Z = torch.softmax(model(grid), 1)[:, 1].numpy().reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.contourf(gx, gy, Z, levels=[0, 0.5, 1], colors=["#dbe9f6", "#fde0dd"], alpha=0.8)
    ax.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1)
    flipped = (model(x_f).argmax(1) != y).numpy()
    sel = np.flatnonzero(flipped)[:40]
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", s=8, alpha=0.5, label="clean")
    for i in sel:
        ax.annotate("", xy=x_f[i].numpy(), xytext=x[i].numpy(), arrowprops=dict(arrowstyle="->", color="k", lw=0.8))
    ax.scatter(x_f[sel, 0], x_f[sel, 1], marker="x", color="k", s=30, label=f"FGSM (eps={eps}) flipped")
    ax.set(xlim=(0, 1), ylim=(0, 1), title=f"accuracy: clean {accs[0]:.2f}  |  FGSM {accs[1]:.2f}  |  PGD-20 {accs[2]:.2f}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
