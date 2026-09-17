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

# ---- right: a real experiment ------------------------------------------------
# A task where dropout genuinely helps: 100-dim inputs of which only 10 carry signal,
# 300 training points, a deliberately over-parameterised 100-64-64-2 MLP.
def make_sparse(n, d=100, k=10, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))                                  # (n, d)
    w = np.zeros(d)                                              # (d,)
    w[:k] = rng.choice([-1.0, 1.0], k)                           # only k features matter
    logit = x @ w / np.sqrt(k) * 2.0                             # (n,)
    y = (logit + rng.normal(scale=noise, size=n) > 0).astype(int)  # (n,)
    return x, y


N_TR = 300
x_all, y_all = make_sparse(N_TR + 1000)
x_tr, y_tr, x_va, y_va = x_all[:N_TR], y_all[:N_TR], x_all[N_TR:], y_all[N_TR:]


def run(p_drop, epochs=400, lr=0.05, width=64, seed=0):
    rng = np.random.default_rng(seed)
    l1 = Linear(x_tr.shape[1], width, rng)                       # (d, width)
    l2 = Linear(width, width, rng)                               # (width, width)
    l3 = Linear(width, 2, rng)                                   # (width, 2)
    a1, a2 = ReLU(), ReLU()
    d1, d2 = Dropout(p_drop, seed=1), Dropout(p_drop, seed=2)
    loss_fn = CrossEntropyLoss()
    tr, va, acc = [], [], []
    for _ in range(epochs):
        for d in (d1, d2):
            d.training = True
        h = d2.forward(a2.forward(l2.forward(d1.forward(a1.forward(l1.forward(x_tr))))))  # (N_TR, width)
        tr.append(loss_fn.forward(l3.forward(h), y_tr))
        g = l3.backward(loss_fn.backward())                      # (N_TR, 2) -> (N_TR, width)
        g = l1.backward(a1.backward(d1.backward(l2.backward(a2.backward(d2.backward(g))))))
        for lin in (l1, l2, l3):
            lin.W -= lr * lin.dW
            lin.b -= lr * lin.db
        for d in (d1, d2):
            d.training = False                                   # inference: no mask, no rescale
        logits = l3.forward(a2.forward(l2.forward(a1.forward(l1.forward(x_va)))))  # (1000, 2)
        va.append(CrossEntropyLoss().forward(logits, y_va))
        acc.append(float((logits.argmax(1) == y_va).mean()))
    return tr, va, acc


ax = axes[1]
finals = {}
for p, ls in [(0.0, "-"), (0.5, "--")]:
    tr, va, acc = run(p)
    finals[p] = (va[-1], acc[-1])
    ax.plot(tr, ls=ls, color="C0", label=f"train, p={p}")
    ax.plot(va, ls=ls, color="C1", label=f"val, p={p}")
ax.set_xlabel("epoch (full-batch GD; 300 train / 1000 val, 100 features of which 10 matter)")
ax.set_ylabel("cross-entropy")
ax.set_title("100-64-64-2 MLP: p=0 overfits (val {:.2f}, acc {:.2f});\n"
             "p=0.5 does not (val {:.2f}, acc {:.2f})".format(
                 finals[0.0][0], finals[0.0][1], finals[0.5][0], finals[0.5][1]), fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_dropout.png", dpi=150, bbox_inches="tight", facecolor="white")
