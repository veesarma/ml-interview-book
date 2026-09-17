"""Two measured facts about the NumPy autograd engine of Part III chapter 3:
the cost of a reverse pass relative to the forward, and its agreement with
torch.autograd on random graphs.
-> docs/assets/figures/part03_autograd_modes.png
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mlbook.nn.autograd import Tensor, cross_entropy

rng = np.random.default_rng(0)

# --- panel 1: forward vs backward wall clock for a chain of matmuls -----------
widths = [32, 64, 128, 256, 512]
fwd_ms, bwd_ms = [], []
for d in widths:
    x = Tensor(rng.normal(size=(64, d)) / np.sqrt(d))          # (64, d)
    ws = [Tensor(rng.normal(size=(d, d)) / np.sqrt(d)) for _ in range(8)]  # 8 x (d, d)
    t0 = time.perf_counter()
    for _ in range(5):
        h = x                                                   # (64, d)
        for w in ws:
            h = (h @ w).relu()                                  # (64, d)
        loss = h.sum()                                          # scalar
    fwd_ms.append((time.perf_counter() - t0) / 5 * 1e3)
    t0 = time.perf_counter()
    for _ in range(5):
        for w in ws:
            w.zero_grad()
        x.zero_grad()
        h = x
        for w in ws:
            h = (h @ w).relu()
        loss = h.sum()
        loss.backward()
    bwd_ms.append((time.perf_counter() - t0) / 5 * 1e3)

ratio = [b / f for f, b in zip(fwd_ms, bwd_ms)]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor="white")
ax = axes[0]
ax.plot(widths, fwd_ms, "o-", label="forward only")
ax.plot(widths, bwd_ms, "s-", label="forward + backward")
ax.set_xscale("log", base=2)
ax.set_xlabel("layer width d (8 layers, batch 64)")
ax.set_ylabel("time per step (ms)")
ax.set_title("A full step costs about {:.1f}x the forward\n(2 extra GEMMs per matmul)".format(np.mean(ratio)), fontsize=9)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
for x_, r in zip(widths, ratio):
    ax.annotate(f"{r:.1f}x", (x_, max(fwd_ms[widths.index(x_)], 0)), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=7, color="#374151")

# --- panel 2: agreement with torch.autograd over random graphs ---------------
errs = []
for trial in range(200):
    n, d, k = 5, 4, 3
    arrays = [rng.normal(size=s) for s in [(n, d), (d, d), (d,), (d, k), (k,)]]
    y = rng.integers(0, k, n)                                   # (n,)
    ts = [Tensor(a, requires_grad=True) for a in arrays]
    tt = [torch.tensor(a, requires_grad=True) for a in arrays]
    xn, w1n, b1n, w2n, b2n = ts
    xt, w1t, b1t, w2t, b2t = tt
    act = trial % 3
    hn = (xn @ w1n + b1n)                                       # (n, d)
    ht = (xt @ w1t + b1t)                                       # (n, d)
    hn, ht = [(hn.relu(), torch.relu(ht)), (hn.tanh(), torch.tanh(ht)),
              (hn.sigmoid(), torch.sigmoid(ht))][act]
    cross_entropy(hn @ w2n + b2n, y).backward()                 # scalar
    torch.nn.functional.cross_entropy(ht @ w2t + b2t, torch.tensor(y)).backward()
    errs.append(max(float(np.max(np.abs(a.grad - b.grad.numpy()))) for a, b in zip(ts, tt)))

ax = axes[1]
ax.hist(np.log10(np.maximum(errs, 1e-20)), bins=30, color="C2", edgecolor="white")
ax.axvline(np.log10(1e-9), color="k", ls="--", lw=1)
ax.text(np.log10(1e-9), ax.get_ylim()[1] * 0.92, " test tolerance 1e-9", fontsize=8, va="top")
ax.set_xlabel("log10 max |grad(engine) - grad(torch.autograd)|")
ax.set_ylabel("random graphs (of 200)")
ax.set_title("Agreement with torch.autograd:\nworst case {:.0e} over 200 random 2-layer graphs".format(max(errs)), fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_autograd_modes.png", dpi=150, bbox_inches="tight", facecolor="white")
print("mean backward/forward ratio", np.mean(ratio), "max err", max(errs))
