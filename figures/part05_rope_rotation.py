"""RoPE: (left) one 2-D pair of q rotated by its position; the angle between R_m q and R_n k
is the content angle plus (n - m) theta. (right) rotation angle per position for several
frequency bands theta_i = 10000^(-2i/d). Writes docs/assets/figures/part05_rope_rotation.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_rope_rotation.png"


def rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    q = np.array([1.0, 0.25])
    k = np.array([0.6, 0.8])
    theta = 0.35  # one frequency band
    for m, style in ((0, "-"), (3, "--")):
        qm = rot(m * theta) @ q
        ax1.annotate("", xy=qm, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colors[0], lw=2, linestyle=style))
        ax1.text(*(qm * 1.08), f"$R_{{{m}}}q$", color=colors[0], fontsize=10)
    for n, style in ((1, "-"), (4, "--")):
        kn = rot(n * theta) @ k
        ax1.annotate("", xy=kn, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colors[1], lw=2, linestyle=style))
        ax1.text(*(kn * 1.08), f"$R_{{{n}}}k$", color=colors[1], fontsize=10)
    circle = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(circle), np.sin(circle), color="#cccccc", lw=0.8)
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_title("One 2-D pair: (m, n) = (0, 1) and (3, 4) give the same angle,\nso the same dot product", fontsize=9.5, loc="left")
    ax1.axhline(0, color="#dddddd", lw=0.6)
    ax1.axvline(0, color="#dddddd", lw=0.6)
    d = 64
    pos = np.arange(0, 64)
    for i, c in zip((0, 4, 12, 24, 31), colors):
        theta_i = 10000 ** (-2 * i / d)
        ax2.plot(pos, np.sin(pos * theta_i), color=c, label=rf"pair {i}: $\theta_i=10000^{{-2\cdot{i}/64}}$ = {theta_i:.2g}")
    ax2.set_xlabel("position m")
    ax2.set_ylabel(r"$\sin(m\,\theta_i)$")
    ax2.set_title("Frequency bands (d_head = 64): pair 0 spins every ~6 tokens,\npair 31 barely moves within 64 tokens", fontsize=9.5, loc="left")
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.grid(alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
