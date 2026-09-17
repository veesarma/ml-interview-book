"""Activation std per layer under different initialisers (ReLU and tanh nets).
-> docs/assets/figures/part03_init_variance.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.nn.init import activation_std_by_depth, kaiming_normal, lecun_normal, orthogonal, xavier_normal


def small_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, 0.01, size=(fan_in, fan_out))


def big_normal(fan_in, fan_out, rng):
    return rng.normal(0.0, 0.2, size=(fan_in, fan_out))


relu = lambda z: np.maximum(z, 0.0)
inits = [("N(0, 0.01^2)", small_normal), ("N(0, 0.2^2)", big_normal), ("LeCun 1/n", lecun_normal),
         ("Xavier 2/(n_in+n_out)", xavier_normal), ("He 2/n", kaiming_normal), ("orthogonal x sqrt2", lambda a, b, r: orthogonal(a, b, r, gain=np.sqrt(2)))]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="white")
for name, init in inits:
    axes[0].semilogy(np.arange(1, 31), activation_std_by_depth(init, relu, width=256, depth=30), label=name, lw=1.6)
    axes[1].semilogy(np.arange(1, 31), activation_std_by_depth(init, np.tanh, width=256, depth=30), label=name, lw=1.6)
axes[0].set_title("ReLU net, width 256: std of activations per layer")
axes[1].set_title("tanh net, width 256")
for ax in axes:
    ax.set_xlabel("layer")
    ax.set_ylabel("std(h)")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(1e-8, 1e3)
axes[0].legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_init_variance.png", dpi=150, bbox_inches="tight", facecolor="white")
