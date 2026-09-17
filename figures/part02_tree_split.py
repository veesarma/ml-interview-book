"""A depth-2 decision tree's axis-aligned partition of the plane, and the
information-gain curve for the root split."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, ORANGE, RED, save, style
from mlbook.classical.decision_tree import DecisionTree, class_proportions, entropy

style()
rng = np.random.default_rng(0)
N = 300
X = rng.uniform(0, 1, (N, 2))  # (N, 2)
y = ((X[:, 0] > 0.55) & (X[:, 1] > 0.35) | (X[:, 0] < 0.2)).astype(int)  # (N,)
tree = DecisionTree(max_depth=2, criterion="entropy").fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
ax = axes[0]
g = np.linspace(0, 1, 250)
G1, G2 = np.meshgrid(g, g)
Z = tree.predict(np.stack([G1.ravel(), G2.ravel()], axis=1)).reshape(G1.shape)  # (250, 250)
ax.contourf(G1, G2, Z, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.18)
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, color=BLUE, label="class 0", edgecolor="white", linewidth=0.3)
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, color=ORANGE, label="class 1", edgecolor="white", linewidth=0.3)
root = tree.root
ax.axvline(root.threshold, color="black", lw=1.5) if root.feature == 0 else ax.axhline(root.threshold, color="black", lw=1.5)
for child, side in [(root.left, "left"), (root.right, "right")]:
    if child.is_leaf:
        continue
    if root.feature == 0:
        xr = (0, root.threshold) if side == "left" else (root.threshold, 1)
        if child.feature == 1:
            ax.plot(xr, [child.threshold] * 2, color="black", lw=1, ls="--")
        else:
            ax.plot([child.threshold] * 2, [0, 1], color="black", lw=1, ls="--")
ax.set_title(f"Depth-2 tree: root split $x_{root.feature + 1}$ ≤ {root.threshold:.2f} (solid)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend(loc="lower right")

ax = axes[1]
parent = entropy(class_proportions(y, 2))
for j, color in [(0, BLUE), (1, ORANGE)]:
    ts = np.linspace(0.02, 0.98, 200)
    gains = []
    for t in ts:
        m = X[:, j] <= t
        child = m.mean() * entropy(class_proportions(y[m], 2)) + (~m).mean() * entropy(class_proportions(y[~m], 2))
        gains.append(parent - child)
    ax.plot(ts, gains, color=color, lw=1.8, label=f"split on $x_{j + 1}$")
ax.axvline(root.threshold, color=RED, lw=1, ls=":")
ax.annotate("chosen root split", (root.threshold, max(gains) * 0.9), xytext=(8, 0), textcoords="offset points", color=RED)
ax.set_xlabel("threshold t"); ax.set_ylabel("information gain (bits)")
ax.set_title("Root split search: gain vs threshold")
ax.legend()
save(fig, "part02_tree_split")
