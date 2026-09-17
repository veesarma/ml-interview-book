"""Near-duplicate detection: the LSH banding S-curve, and how MinHash estimation error
falls with the number of permutations.

Both panels are computed from mlbook.llm.data_dedup, not from published figures, so the
chapter's claims and the code agree by construction.

Writes docs/assets/figures/part06_dedup_lsh.png.  Run from the repository root.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlbook.llm.data_dedup import MinHash, estimate_jaccard, jaccard, shingles  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part06_dedup_lsh.png"

# (num_perm, bands) configurations; rows = num_perm // bands.
CONFIGS = [(128, 32), (128, 16), (128, 8), (256, 16)]


def _perturbed(words: list[str], frac: float, rng: np.random.Generator) -> str:
    """Replace `frac` of the words with filler, giving a controlled Jaccard target."""
    out = list(words)
    n_change = int(len(words) * frac)
    idx = rng.choice(len(words), size=n_change, replace=False)
    for i in idx:
        out[i] = f"zzz{i}"
    return " ".join(out)


def main() -> None:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")

    # Panel 1: candidate probability 1 - (1 - J^r)^b, the banding S-curve.
    J = np.linspace(0, 1, 500)  # (500,) Jaccard similarity
    for i, (num_perm, bands) in enumerate(CONFIGS):
        rows = num_perm // bands
        p_candidate = 1.0 - (1.0 - J**rows) ** bands  # (500,)
        thr = (1.0 / bands) ** (1.0 / rows)
        ax1.plot(J, p_candidate, color=colors[i], linewidth=2,
                 label=f"{num_perm} perms, b={bands}, r={rows}  (thr {thr:.2f})")
        ax1.scatter([thr], [0.5], color=colors[i], s=22, zorder=5)
    ax1.axhline(0.5, color="#bbbbbb", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("true Jaccard similarity $J$")
    ax1.set_ylabel("P(pair becomes a candidate)")
    ax1.set_title("LSH banding: $1-(1-J^r)^b$", fontsize=10, loc="left")
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax1.grid(color="#eeeeee", linewidth=0.5)

    # Panel 2: MinHash estimation error vs number of permutations (empirical vs theory).
    rng = np.random.default_rng(0)
    base_words = [f"w{i}" for i in range(400)]
    base_text = " ".join(base_words)
    base_sh = shingles(base_text, 5)
    perms = [16, 32, 64, 128, 256, 512]
    errors, theory = [], []
    # Average over several hash seeds and several perturbation levels so the trend,
    # rather than the sampling noise of one pair, is what the panel shows.
    for k in perms:
        errs, js = [], []
        for seed in range(6):
            mh = MinHash(num_perm=k, seed=seed)
            sig_a = mh.signature(base_sh)
            for frac in (0.05, 0.1, 0.2, 0.3):
                other = shingles(_perturbed(base_words, frac, rng), 5)
                exact = jaccard(base_sh, other)
                est = estimate_jaccard(sig_a, mh.signature(other))
                errs.append(abs(est - exact))
                js.append(exact)
        errors.append(np.mean(errs))
        # E|X - J| for X ~ Normal is sqrt(2/pi) * sigma, so compare like with like.
        theory.append(np.sqrt(2 / np.pi) * np.mean([np.sqrt(j * (1 - j) / k) for j in js]))
    ax2.plot(perms, errors, marker="o", color=colors[0], linewidth=2, label="measured |estimate − exact|")
    ax2.plot(perms, theory, marker="o", color=colors[3], linewidth=2, linestyle="--",
             label=r"theory $\sqrt{2/\pi}\,\sqrt{J(1-J)/k}$")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("number of permutations $k$")
    ax2.set_ylabel("absolute error of the Jaccard estimate")
    ax2.set_title("MinHash accuracy: error falls as $1/\\sqrt{k}$", fontsize=10, loc="left")
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(color="#eeeeee", linewidth=0.5, which="both")

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
