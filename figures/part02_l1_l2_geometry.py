"""L1 vs L2 regularisation geometry: the loss contours meet the L1 ball at a corner
(sparse solution) but the L2 ball at a generic point."""
import numpy as np
import matplotlib.pyplot as plt

from _common_part02 import BLUE, ORANGE, GREY, RED, save, style

style()
# A least-squares loss with an elongated, tilted contour set.
A = np.array([[3.0, 1.2], [1.2, 1.0]])  # (2, 2) Hessian of the quadratic
w_ols = np.array([1.6, 1.1])  # (2,) unconstrained minimiser
g = np.linspace(-2.2, 2.6, 400)
W1, W2 = np.meshgrid(g, g)  # (400, 400) each
D = np.stack([W1 - w_ols[0], W2 - w_ols[1]], axis=-1)  # (400, 400, 2)
loss = np.einsum("...i,ij,...j->...", D, A, D)  # (400, 400) quadratic form (Σ_i Σ_j d_i A_ij d_j)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharey=True)
for ax, name in zip(axes, ["L1 (Lasso)", "L2 (Ridge)"]):
    ax.contour(W1, W2, loss, levels=np.linspace(0.3, 12, 9), colors=GREY, linewidths=0.8)
    t = 1.0  # radius of the constraint ball
    if name.startswith("L1"):
        ball = np.array([[t, 0], [0, t], [-t, 0], [0, -t], [t, 0]])  # (5, 2) diamond
        ax.plot(ball[:, 0], ball[:, 1], color=BLUE, lw=2)
        ax.fill(ball[:, 0], ball[:, 1], color=BLUE, alpha=0.12)
        # find the touching point numerically on the ball's boundary
        cand = np.array([[t * (1 - s), t * s] for s in np.linspace(0, 1, 2001)])  # (2001, 2) first-quadrant edge
    else:
        th = np.linspace(0, 2 * np.pi, 300)
        ax.plot(t * np.cos(th), t * np.sin(th), color=ORANGE, lw=2)
        ax.fill(t * np.cos(th), t * np.sin(th), color=ORANGE, alpha=0.12)
        cand = np.stack([t * np.cos(th), t * np.sin(th)], axis=1)  # (300, 2)
    dc = cand - w_ols  # (n, 2)
    vals = np.einsum("ni,ij,nj->n", dc, A, dc)  # (n,)
    w_hat = cand[vals.argmin()]
    ax.plot(*w_ols, "o", color=RED, ms=7)
    ax.annotate("OLS", w_ols, xytext=(6, 4), textcoords="offset points", color=RED)
    ax.plot(*w_hat, "s", color="black", ms=7)
    ax.annotate(f"w* = ({w_hat[0]:.2f}, {w_hat[1]:.2f})", w_hat, xytext=(8, -14), textcoords="offset points")
    ax.set_title(name)
    ax.set_xlabel("$w_1$")
    ax.set_aspect("equal")
axes[0].set_ylabel("$w_2$")
fig.suptitle("Loss contours meet the L1 ball at a corner (w₂ = 0) and the L2 ball at an interior boundary point", fontsize=10)
save(fig, "part02_l1_l2_geometry")
