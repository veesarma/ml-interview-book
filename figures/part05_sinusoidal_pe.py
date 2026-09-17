"""The sinusoidal positional-encoding matrix (T=100, d=64) and the dot product PE[t] . PE[t + k].
Writes docs/assets/figures/part05_sinusoidal_pe.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlbook.transformer.positional import sinusoidal_positional_encoding  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_sinusoidal_pe.png"


def main() -> None:
    T, d = 100, 64
    pe = sinusoidal_positional_encoding(T, d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1]})
    im = ax1.imshow(pe.numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ax1.set_xlabel("dimension (even = sin, odd = cos; frequency falls left to right)")
    ax1.set_ylabel("position t")
    ax1.set_title("PE[t, :] for T = 100, d = 64", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    sims = (pe @ pe.T).numpy()
    for t in (0, 20, 50):
        ax2.plot(range(T), sims[t], label=f"t = {t}")
    ax2.set_xlabel("position s")
    ax2.set_ylabel("PE[t] . PE[s]")
    ax2.set_title("Similarity depends on |t - s| and decays with it", loc="left", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
