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

# --- panel 1: cost of a full step relative to the forward pass ----------------
def build(x, ws):
    """Forward through 8 relu(h @ w) layers; returns the scalar loss node."""
    h = x                                                       # (64, d)
    for w in ws:
        h = (h @ w).relu()                                      # (64, d)
    return h.sum()                                              # scalar


def median_ms(fn, repeats=7):
    fn()                                                        # warm up (BLAS threads, caches)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


widths = [64, 128, 256, 512, 1024]
ratios, fwd_ms = [], []
for d in widths:
    x = Tensor(rng.normal(size=(64, d)) / np.sqrt(d))           # (64, d)
    ws = [Tensor(rng.normal(size=(d, d)) / np.sqrt(d)) for _ in range(8)]   # 8 x (d, d)

    def fwd(x=x, ws=ws):
        build(x, ws)

    def step(x=x, ws=ws):
        for w in ws:
            w.zero_grad()
        x.zero_grad()
        build(x, ws).backward()

    f, s_ = median_ms(fwd), median_ms(step)
    fwd_ms.append(f)
    ratios.append(s_ / f)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor="white")
ax = axes[0]
ax.bar([str(w) for w in widths], ratios, color="C0", width=0.6)
ax.axhline(3.0, color="k", ls="--", lw=1)
ax.text(len(widths) - 0.4, 3.05, "3x (2 backward GEMMs\nper forward GEMM)", fontsize=8, ha="right")
for i, (r, f) in enumerate(zip(ratios, fwd_ms)):
    ax.text(i, r + 0.06, f"{r:.1f}x", ha="center", fontsize=8)
    ax.text(i, 0.12, f"fwd {f:.0f} ms", ha="center", fontsize=7, color="white")
ax.set_ylim(0, max(3.4, max(ratios) + 0.5))
ax.set_xlabel("layer width d (8 layers, batch 64, median of 7 runs)")
ax.set_ylabel("(forward + backward) / forward")
ax.set_title("Cost of a full step relative to the forward pass;\nPython graph overhead dominates at small widths", fontsize=9)
ax.grid(alpha=0.3, axis="y")

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
