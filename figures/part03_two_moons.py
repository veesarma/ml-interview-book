"""NumPy MLP trained with hand-written backprop on two moons.
-> docs/assets/figures/part03_two_moons.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.nn.mlp import MLP, accuracy, make_two_moons, train_classifier

x, y = make_two_moons(n=400, noise=0.1, seed=0)
model = MLP([2, 32, 32, 2], activation="relu", seed=0)
hist = train_classifier(model, x, y, epochs=150, lr=0.1, batch_size=32)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="white")
axes[0].plot(hist)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("train cross-entropy")
axes[0].set_title(f"loss (final accuracy {accuracy(model, x, y):.3f})")
axes[0].grid(alpha=0.3)

xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.7, 300))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (90000, 2)
logits = model.forward(grid)  # (90000, 2)
p1 = np.exp(logits - logits.max(1, keepdims=True))
p1 = (p1 / p1.sum(1, keepdims=True))[:, 1].reshape(xx.shape)
axes[1].contourf(xx, yy, p1, levels=20, cmap="RdBu_r", alpha=0.6)
axes[1].contour(xx, yy, p1, levels=[0.5], colors="k", linewidths=1)
axes[1].scatter(x[:, 0], x[:, 1], c=y, cmap="RdBu_r", s=10, edgecolors="k", linewidths=0.3)
axes[1].set_title("decision boundary, MLP 2-32-32-2 (NumPy backprop)")
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_two_moons.png", dpi=150, bbox_inches="tight", facecolor="white")
