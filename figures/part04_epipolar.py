"""Two synthetic views of a 3-D point set with epipolar lines from the 8-point fundamental matrix."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.geometry import camera as C
from mlbook.geometry import epipolar as E

OUT = "docs/assets/figures/part04_epipolar.png"


def main():
    rng = np.random.default_rng(0)
    K = C.intrinsics(500.0, 500.0, 320.0, 240.0)
    R1, t1 = C.look_at(np.array([0.0, -5.0, 1.5]), np.array([0.0, 0.0, 1.0]))
    R2, t2 = C.look_at(np.array([2.5, -4.3, 1.2]), np.array([0.0, 0.0, 1.0]))
    X = rng.uniform(-1.2, 1.2, size=(40, 3)) + np.array([0.0, 0.0, 1.0])
    p1, _ = C.project(X, K, R1, t1)
    p2, _ = C.project(X, K, R2, t2)
    F = E.eight_point(p1, p2)
    lines = E.epipolar_lines(F, p1[:6])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    for ax, p, title in ((ax1, p1, "image 1"), (ax2, p2, "image 2")):
        ax.scatter(p[:, 0], p[:, 1], s=12, c="0.5")
        ax.scatter(p[:6, 0], p[:6, 1], s=30, c=[f"C{i}" for i in range(6)])
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_aspect("equal")
        ax.set_title(title)
    xs = np.array([0.0, 640.0])
    for i, (a, b, c) in enumerate(lines):
        ax2.plot(xs, -(a * xs + c) / b, color=f"C{i}", lw=1)
    e1, e2 = E.epipoles(F)
    ax2.set_title(f"image 2: epipolar lines l' = F p  (epipole at {e2[0]:.0f}, {e2[1]:.0f})")
    fig.suptitle("Each point in image 1 constrains its match in image 2 to a line — the search is 1-D")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
