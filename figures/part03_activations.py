"""Activation functions and their derivatives -> docs/assets/figures/part03_activations.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.nn.layers import GELU, ReLU, Sigmoid, SiLU, Tanh

z = np.linspace(-5, 5, 801)
layers = [("ReLU", ReLU()), ("GELU", GELU()), ("SiLU", SiLU()), ("sigmoid", Sigmoid()), ("tanh", Tanh())]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="white")
for name, layer in layers:
    f = layer.forward(z)
    df = layer.backward(np.ones_like(z))
    axes[0].plot(z, f, label=name, lw=1.8)
    axes[1].plot(z, df, label=name, lw=1.8)
axes[0].set_title("f(z)")
axes[1].set_title("f'(z)  (what backprop multiplies by)")
for ax in axes:
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("z")
    ax.grid(alpha=0.3)
axes[0].set_ylim(-1.2, 3)
axes[1].set_ylim(-0.3, 1.3)
axes[1].legend(loc="upper left", fontsize=9)
axes[1].annotate("sigmoid' <= 1/4: vanishing\ngradient when stacked", xy=(0, 0.25), xytext=(1.2, 0.6),
                 arrowprops=dict(arrowstyle="->"), fontsize=8)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_activations.png", dpi=150, bbox_inches="tight", facecolor="white")
