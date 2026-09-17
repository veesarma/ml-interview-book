"""Computational graph of a 2-layer MLP with forward shapes and backward gradients.
-> docs/assets/figures/part03_computational_graph.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="white")
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.2)
ax.axis("off")

nodes = [  # (x, label, shape)
    (0.9, "X", "(N, d)"),
    (2.5, "Z1 = X W1 + b1", "(N, h)"),
    (4.3, "H = relu(Z1)", "(N, h)"),
    (6.1, "Z2 = H W2 + b2", "(N, K)"),
    (7.9, "P = softmax(Z2)", "(N, K)"),
    (9.7, "L = CE(P, y)", "()"),
]
y_fwd = 2.6
for x, label, shape in nodes:
    ax.add_patch(FancyBboxPatch((x - 0.75, y_fwd - 0.32), 1.5, 0.64, boxstyle="round,pad=0.05",
                                fc="#dbeafe", ec="#1f2937", lw=1))
    ax.text(x, y_fwd + 0.05, label, ha="center", va="center", fontsize=8.5, family="monospace")
    ax.text(x, y_fwd - 0.2, shape, ha="center", va="center", fontsize=7.5, color="#374151")
for (x0, _, _), (x1, _, _) in zip(nodes[:-1], nodes[1:]):
    ax.add_patch(FancyArrowPatch((x0 + 0.78, y_fwd + 0.1), (x1 - 0.78, y_fwd + 0.1),
                                 arrowstyle="-|>", mutation_scale=12, color="#1f2937"))

params = [(2.5, "W1 (d,h)  b1 (h,)"), (6.1, "W2 (h,K)  b2 (K,)")]
for x, label in params:
    ax.add_patch(FancyBboxPatch((x - 0.75, 3.5), 1.5, 0.45, boxstyle="round,pad=0.05", fc="#fef3c7", ec="#92400e"))
    ax.text(x, 3.72, label, ha="center", va="center", fontsize=7.5, family="monospace")
    ax.add_patch(FancyArrowPatch((x, 3.48), (x, y_fwd + 0.36), arrowstyle="-|>", mutation_scale=10, color="#92400e"))

grads = [
    (9.7, "dL/dL = 1"),
    (7.9, "dZ2 = (P - Y)/N\n(N, K)"),
    (6.1, "dH = dZ2 W2^T   (N, h)\ndW2 = H^T dZ2   (h, K)\ndb2 = sum_n dZ2  (K,)"),
    (4.3, "dZ1 = dH * 1[Z1>0]\n(N, h)"),
    (2.5, "dX = dZ1 W1^T   (N, d)\ndW1 = X^T dZ1   (d, h)\ndb1 = sum_n dZ1  (h,)"),
]
y_bwd = 1.0
for x, label in grads:
    ax.add_patch(FancyBboxPatch((x - 0.85, y_bwd - 0.45), 1.7, 0.9, boxstyle="round,pad=0.05", fc="#fee2e2", ec="#991b1b"))
    ax.text(x, y_bwd, label, ha="center", va="center", fontsize=6.8, family="monospace")
for (x1, _), (x0, _) in zip(grads[:-1], grads[1:]):
    ax.add_patch(FancyArrowPatch((x1 - 0.88, y_bwd), (x0 + 0.88, y_bwd), arrowstyle="-|>", mutation_scale=12, color="#991b1b"))
ax.text(0.15, y_fwd + 0.75, "forward", fontsize=9, color="#1f2937", va="center")
ax.text(10.55, 3.72, "cache X, Z1, H, P\nfor backward", fontsize=7.5, color="#374151", va="center", ha="right")
ax.text(0.15, y_bwd + 0.7, "backward: reverse topological order, one vector-Jacobian product per node",
        fontsize=9, color="#991b1b", va="center")
fig.savefig("docs/assets/figures/part03_computational_graph.png", dpi=150, bbox_inches="tight", facecolor="white")
