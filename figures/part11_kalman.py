"""The Kalman filter: how predict inflates the covariance and update shrinks it.

Writes docs/assets/figures/part11_kalman.png
Run from the repository root:  python figures/part11_kalman.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.perception.kalman import KalmanFilter, constant_velocity_model, covariance_ellipse  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part11_kalman.png"


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    F, H = constant_velocity_model(2, dt=1.0)
    Q = np.diag([0.35, 0.35, 0.02, 0.02])
    R = 0.55 * np.eye(2)
    rng = np.random.default_rng(3)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), facecolor="white")

    # ---- left: one predict/update cycle, drawn large ---------------------------------
    ax = axes[0]
    R_demo = 0.22 * np.eye(2)  # a confident sensor, so the update visibly shrinks the ellipse
    kf = KalmanFilter(F, H, Q, R_demo, np.array([0.0, 0.0, 1.6, 0.35]), np.diag([0.55, 0.55, 0.4, 0.4]))
    prior = kf.x[:2].copy()
    e_prior = covariance_ellipse(kf.P[:2, :2], n_std=2.0) + prior
    kf.predict()
    pred = kf.x[:2].copy()
    e_pred = covariance_ellipse(kf.P[:2, :2], n_std=2.0) + pred
    z = pred + np.array([0.85, -0.55])
    kf.update(z)
    post = kf.x[:2].copy()
    e_post = covariance_ellipse(kf.P[:2, :2], n_std=2.0) + post

    ax.plot(*e_prior.T, color=colors[7], lw=1.6, label="posterior at t-1")
    ax.plot(*e_pred.T, color=colors[0], lw=2.0, label=r"prediction: $P^- = FPF^\top + Q$")
    ax.plot(*e_post.T, color=colors[2], lw=2.0, label=r"update: $P = (I-KH)P^-$")
    circ = covariance_ellipse(R_demo, n_std=2.0) + z
    ax.plot(*circ.T, color=colors[3], lw=1.5, ls="--", label="measurement noise R")
    ax.scatter(*prior, color=colors[7], s=30, zorder=5)
    ax.scatter(*pred, color=colors[0], s=45, zorder=5)
    ax.scatter(*z, color=colors[3], s=55, marker="x", zorder=5)
    ax.scatter(*post, color=colors[2], s=55, zorder=5)
    ax.annotate("", xy=pred, xytext=prior, arrowprops=dict(arrowstyle="->", color="#777777"))
    ax.annotate("", xy=post, xytext=pred, arrowprops=dict(arrowstyle="->", color=colors[2], lw=1.4))
    mid = (pred + z) / 2 + np.array([0.0, 0.62])
    ax.text(*mid, "the innovation\n" r"$y = z - H\hat x^-$", fontsize=8, ha="center", color="#333333")
    ax.set_title("one cycle: predict grows the ellipse, update shrinks it", fontsize=10, loc="left")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="lower right")
    ax.set_aspect("equal")
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: a full track, plus the trace of P over time --------------------------
    ax = axes[1]
    kf = KalmanFilter(F, H, Q, R, np.array([0.0, 0.0, 0.0, 0.0]), np.diag([1.0, 1.0, 25.0, 25.0]))
    truth = np.array([0.0, 0.0, 1.6, 0.35])
    tr, zs, xs, traces = [], [], [], []
    for t in range(18):
        truth = F @ truth
        kf.predict()
        if t not in (7, 8, 9):                       # three missed detections: the track coasts
            z = truth[:2] + rng.normal(0, np.sqrt(R[0, 0]), size=2)
            kf.update(z)
            zs.append(z)
        tr.append(truth[:2].copy())
        xs.append(kf.x[:2].copy())
        traces.append(np.trace(kf.P[:2, :2]))
        if t % 3 == 0:
            e = covariance_ellipse(kf.P[:2, :2], n_std=2.0) + kf.x[:2]
            ax.plot(*e.T, color=colors[2], lw=0.9, alpha=0.65)
    tr, zs, xs = np.array(tr), np.array(zs), np.array(xs)
    ax.plot(tr[:, 0], tr[:, 1], color="#666666", lw=1.6, label="true track")
    ax.scatter(zs[:, 0], zs[:, 1], color=colors[3], s=20, marker="x", label="detections")
    ax.plot(xs[:, 0], xs[:, 1], color=colors[2], lw=1.8, label="filter estimate")
    ax.axvspan(tr[7, 0], tr[9, 0], color=colors[1], alpha=0.12)
    ax.text(tr[8, 0], tr[8, 1] + 1.6, "occluded:\nno detections", fontsize=8, ha="center", color="#555555")
    ax.set_title("coasting through an occlusion, 2-sigma ellipses every 3 frames", fontsize=10, loc="left")
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.legend(fontsize=7.6, frameon=False, loc="upper left")
    ax.grid(color="#eeeeee", lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    inset = ax.inset_axes((0.62, 0.08, 0.35, 0.3))
    inset.plot(traces, color=colors[2], lw=1.4)
    inset.axvspan(7, 9, color=colors[1], alpha=0.18)
    inset.set_title(r"tr$(P_{xy})$", fontsize=7.5)
    inset.tick_params(labelsize=6.5)
    inset.grid(color="#eeeeee", lw=0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
