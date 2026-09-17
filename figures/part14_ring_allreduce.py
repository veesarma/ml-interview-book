"""Ring all-reduce on 4 ranks: reduce-scatter (3 steps) then all-gather (3 steps).
Each cell = one chunk on one rank; the number is how many ranks' contributions it holds."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_ring_allreduce.png"


def main() -> None:
    N = 4
    state = np.ones((N, N), dtype=int)  # (rank, chunk): contributions held
    frames = [("start", state.copy())]
    # reduce-scatter: at step s rank r sends chunk (r - s) mod N to rank r+1, which accumulates
    for s in range(N - 1):
        new = state.copy()
        for r in range(N):
            c = (r - s - 1) % N
            new[(r + 1) % N, c] = state[(r + 1) % N, c] + state[r, c]
        state = new
        frames.append((f"reduce-scatter step {s + 1}", state.copy()))
    # all-gather: rank r sends its complete chunk (r + 1 - s) mod N onward
    for s in range(N - 1):
        new = state.copy()
        for r in range(N):
            c = (r + 1 - s) % N
            new[(r + 1) % N, c] = state[r, c]
        state = new
        frames.append((f"all-gather step {s + 1}", state.copy()))
    fig, axes = plt.subplots(1, len(frames), figsize=(2.1 * len(frames), 2.9))
    for ax, (title, st) in zip(axes, frames):
        ax.imshow(st, cmap="Blues", vmin=0, vmax=N)
        for i in range(N):
            for j in range(N):
                ax.text(j, i, str(st[i, j]), ha="center", va="center", fontsize=9, color="white" if st[i, j] >= 3 else "black")
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(N)); ax.set_xticklabels([f"c{j}" for j in range(N)], fontsize=7)
        ax.set_yticks(range(N)); ax.set_yticklabels([f"rank {i}" for i in range(N)], fontsize=7)
    fig.suptitle("Ring all-reduce, N=4: 2(N-1)=6 steps, each moving S/N bytes per rank -> 2(N-1)/N * S bytes on the wire", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
