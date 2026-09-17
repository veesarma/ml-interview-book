"""Part III chapter 3: what the NumPy autograd engine actually does.

Left:  the graph of L = sum(relu(x*w + b)) with every node's forward value and
       backward .grad taken from a real run of mlbook.nn.autograd.
Right: agreement with torch.autograd over 200 random 2-layer graphs.

-> docs/assets/figures/part03_autograd_modes.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from mlbook.nn.autograd import Tensor, _topological_order, cross_entropy

# --- panel 1: a real forward/backward on a five-node graph -------------------
x = Tensor([1.0, -2.0], requires_grad=True)   # (2,)
w = Tensor([0.5, 3.0], requires_grad=True)    # (2,)
b = Tensor([0.5], requires_grad=True)         # (1,)  broadcast over the 2 elements
m = x * w                                     # (2,)
z = m + b                                     # (2,)
h = z.relu()                                  # (2,)
loss = h.sum()                                # scalar
order = _topological_order(loss)              # parents before children
loss.backward()

rank = {id(t): i for i, t in enumerate(order)}
nodes = [  # (tensor, label, column, row)
    (x, "x", 0, 1.9), (w, "w", 0, 0.75), (b, "b", 0, -0.35),
    (m, "m = x * w", 1, 1.3), (z, "z = m + b", 2, 0.75),
    (h, "h = relu(z)", 3, 0.75), (loss, "L = h.sum()", 4, 0.75),
]
edges = [(x, m), (w, m), (m, z), (b, z), (z, h), (h, loss)]
pos = {id(t): (c, r) for t, _, c, r in nodes}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), facecolor="white",
                         gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
for src, dst in edges:
    x0, y0 = pos[id(src)]
    x1, y1 = pos[id(dst)]
    ax.add_patch(FancyArrowPatch((x0 + 0.34, y0), (x1 - 0.34, y1), arrowstyle="-|>",
                                 mutation_scale=11, color="#1d4ed8", lw=1.2,
                                 connectionstyle="arc3,rad=0.04"))
    ax.add_patch(FancyArrowPatch((x1 - 0.34, y1 - 0.13), (x0 + 0.34, y0 - 0.13), arrowstyle="-|>",
                                 mutation_scale=11, color="#b91c1c", lw=1.2,
                                 connectionstyle="arc3,rad=0.04"))
for t, label, c, r in nodes:
    leaf = t.requires_grad
    ax.add_patch(FancyBboxPatch((c - 0.34, r - 0.26), 0.68, 0.52, boxstyle="round,pad=0.03",
                                fc="#fef3c7" if leaf else "#dbeafe",
                                ec="#92400e" if leaf else "#1e3a8a", lw=1.1))
    ax.text(c, r + 0.13, label, ha="center", fontsize=8, family="monospace")
    val = np.array2string(np.atleast_1d(t.data), precision=1, separator=",")
    grd = np.array2string(np.atleast_1d(t.grad), precision=1, separator=",")
    ax.text(c, r - 0.02, val, ha="center", fontsize=7, color="#1d4ed8")
    ax.text(c, r - 0.16, grd, ha="center", fontsize=7, color="#b91c1c")
    ax.text(c - 0.31, r + 0.31, f"#{rank[id(t)]}", fontsize=6.5, color="#374151")
ax.text(-0.45, 2.45, "blue = forward value and order  #i;   red = .grad after loss.backward()",
        fontsize=8, color="#374151")
ax.text(-0.45, -1.15, "relu kills element 1, so its gradient is 0 all the way back to x and w;\n"
                      "b was broadcast over 2 elements, so its gradient is the sum of both.",
        fontsize=8, color="#374151")
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-1.6, 2.6)
ax.axis("off")
ax.set_title("One run of the engine: values forward, gradients back", fontsize=9)

# --- panel 2: agreement with torch.autograd over random graphs ---------------
rng = np.random.default_rng(0)
errs = []
for trial in range(200):
    n, d, k = 5, 4, 3
    arrays = [rng.normal(size=s) for s in [(n, d), (d, d), (d,), (d, k), (k,)]]
    y = rng.integers(0, k, n)                                   # (n,)
    ts = [Tensor(a, requires_grad=True) for a in arrays]
    tt = [torch.tensor(a, requires_grad=True) for a in arrays]
    xn, w1n, b1n, w2n, b2n = ts
    xt, w1t, b1t, w2t, b2t = tt
    pre_n, pre_t = xn @ w1n + b1n, xt @ w1t + b1t               # (n, d)
    hn, ht = [(pre_n.relu(), torch.relu(pre_t)), (pre_n.tanh(), torch.tanh(pre_t)),
              (pre_n.sigmoid(), torch.sigmoid(pre_t))][trial % 3]
    cross_entropy(hn @ w2n + b2n, y).backward()                 # scalar
    torch.nn.functional.cross_entropy(ht @ w2t + b2t, torch.tensor(y)).backward()
    errs.append(max(float(np.max(np.abs(a.grad - b.grad.numpy()))) for a, b in zip(ts, tt)))

ax = axes[1]
ax.hist(np.log10(np.maximum(errs, 1e-20)), bins=28, color="C2", edgecolor="white")
ax.axvline(-9, color="k", ls="--", lw=1)
ax.text(-8.8, ax.get_ylim()[1] * 0.9, "test tolerance 1e-9", fontsize=8, va="top")
ax.set_xlim(-20, -7)
ax.set_xlabel("log10 max |grad(engine) - grad(torch.autograd)|")
ax.set_ylabel("random graphs (of 200)")
ax.set_title("Worst disagreement with torch.autograd over 200\nrandom 2-layer graphs: {:.0e}".format(max(errs)), fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_autograd_modes.png", dpi=150, bbox_inches="tight", facecolor="white")
print("max err", max(errs), "| grads:", x.grad, w.grad, b.grad)
