"""Dropout: (left) three thinned sub-networks sampled from one MLP, (right) a real
train/val experiment on a small noisy two-moons set with and without dropout.
-> docs/assets/figures/part03_dropout.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.nn.layers import Linear, ReLU
from mlbook.nn.losses import CrossEntropyLoss
from mlbook.nn.mlp import make_two_moons
from mlbook.nn.regularization import Dropout

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor="white", gridspec_kw={"width_ratios": [1.1, 1]})

# ---- left: thinned networks --------------------------------------------------
rng = np.random.default_rng(3)
layer_sizes = [3, 6, 6, 2]
xs = np.linspace(0.05, 0.95, len(layer_sizes))
ax = axes[0]
ax.axis("off")
for k in range(3):
    y_off = -k * 1.25
    masks = [np.ones(layer_sizes[0], bool)] + [rng.random(s) > 0.5 for s in layer_sizes[1:-1]] + [np.ones(layer_sizes[-1], bool)]
    coords = []
    for li, size in enumerate(layer_sizes):
        ys = np.linspace(0, 1, size) if size > 1 else np.array([0.5])
        coords.append([(xs[li], y + y_off) for y in ys])
    for li in range(len(layer_sizes) - 1):
        for i, (x0, y0) in enumerate(coords[li]):
            for j, (x1, y1) in enumerate(coords[li + 1]):
                if masks[li][i] and masks[li + 1][j]:
                    ax.plot([x0, x1], [y0, y1], color="#9ca3af", lw=0.5, zorder=1)
    for li, pts in enumerate(coords):
        for i, (x0, y0) in enumerate(pts):
            alive = masks[li][i]
            ax.scatter([x0], [y0], s=60, c="#2563eb" if alive else "white", edgecolors="#1f2937", zorder=2, linewidths=0.8)
    ax.text(-0.02, 0.5 + y_off, f"mask {k+1}", fontsize=8, ha="right", va="center")
ax.set_title("each minibatch trains a different thinned network\n(hollow = dropped, p = 0.5); inference averages them", fontsize=9)
ax.set_xlim(-0.15, 1.05)

# ---- right: a real experiment ---------------------------------------------------
x_all, y_all = make_two_moons(n=120, noise=0.35, seed=1)
x_tr, y_tr, x_va, y_va = x_all[:40], y_all[:40], x_all[40:], y_all[40:]


def run(p_drop, epochs=600, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    l1, l2, l3 = Linear(2, 128, rng), Linear(128, 128, rng), Linear(128, 2, rng)
    a1, a2 = ReLU(), ReLU()
    d1, d2 = Dropout(p_drop, seed=1), Dropout(p_drop, seed=2)
    loss_fn = CrossEntropyLoss()
    tr, va = [], []
    for _ in range(epochs):
        for d in (d1, d2):
            d.training = True
        h = d2.forward(a2.forward(l2.forward(d1.forward(a1.forward(l1.forward(x_tr))))))
        tr.append(loss_fn.forward(l3.forward(h), y_tr))
        g = l3.backward(loss_fn.backward())
        g = l1.backward(a1.backward(d1.backward(l2.backward(a2.backward(d2.backward(g))))))
        for lin in (l1, l2, l3):
            lin.W -= lr * lin.dW
            lin.b -= lr * lin.db
        for d in (d1, d2):
            d.training = False
        h = a2.forward(l2.forward(a1.forward(l1.forward(x_va))))
        va.append(CrossEntropyLoss().forward(l3.forward(h), y_va))
    return tr, va


ax = axes[1]
for p, ls in [(0.0, "-"), (0.5, "--")]:
    tr, va = run(p)
    ax.plot(tr, ls=ls, color="C0", label=f"train, p={p}")
    ax.plot(va, ls=ls, color="C1", label=f"val, p={p}")
ax.set_xlabel("epoch (full-batch GD, 40 train / 80 val points, noise 0.35)")
ax.set_ylabel("cross-entropy")
ax.set_title("2-128-128-2 MLP: dropout closes the train/val gap", fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_dropout.png", dpi=150, bbox_inches="tight", facecolor="white")
