"""A/B testing vs interleaving: statistical power vs sample size (synthetic).

Reproduces the *argument* of Netflix's 2017 Tech Blog post "Innovating Faster on
Personalization Algorithms at Netflix Using Interleaving": interleaving is a paired,
within-user comparison, so it needs far fewer users than a between-user A/B test on a
heavy-tailed engagement metric. The simulation and its numbers are ours, not Netflix's.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part18_consumer_interleaving_power.png"


def ab_power(n: int, lift: float, trials: int, rng: np.random.Generator) -> float:
    """Power of a two-sample z-test on a lognormal engagement metric (per-user hours)."""
    hits = 0
    for _ in range(trials):
        a = rng.lognormal(mean=0.0, sigma=1.2, size=n)               # (n,)
        b = rng.lognormal(mean=np.log(1.0 + lift), sigma=1.2, size=n)  # (n,)
        se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
        hits += abs(b.mean() - a.mean()) / se > 1.96
    return hits / trials


def interleaving_power(n: int, pref: float, trials: int, rng: np.random.Generator) -> float:
    """Power of a binomial test on per-user preference (P(prefer B) = 0.5 + pref)."""
    hits = 0
    for _ in range(trials):
        wins = rng.binomial(n, 0.5 + pref)
        z = (wins - 0.5 * n) / np.sqrt(0.25 * n)
        hits += abs(z) > 1.96
    return hits / trials


def main() -> None:
    rng = np.random.default_rng(1)
    ns = np.unique(np.logspace(2, 5.3, 18).astype(int))
    ab = [ab_power(int(n), 0.02, 300, rng) for n in ns]
    il = [interleaving_power(int(n), 0.02, 300, rng) for n in ns]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(ns, ab, lw=2, color="C0", marker="o", ms=4, label="A/B test on hours watched (2% mean lift, heavy-tailed)")
    ax.plot(ns, il, lw=2, color="C1", marker="s", ms=4, label="interleaving preference test (52% prefer B)")
    ax.axhline(0.8, color="0.5", lw=1, ls="--")
    ax.text(ns[0], 0.815, "80% power", fontsize=8, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("users in the experiment")
    ax.set_ylabel("power (P[detect the true difference])")
    ax.set_title("Why interleaving needs fewer users than A/B (synthetic simulation)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
