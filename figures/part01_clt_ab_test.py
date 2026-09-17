"""CLT in action and the power curve of a two-proportion A/B test."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.math.probability import normal_cdf
from mlbook.math.stats import clt_sample_means, z_critical

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_clt_ab_test.png"


def main() -> None:
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    sampler = lambda size: rng.exponential(1.0, size)  # noqa: E731  (skewed source law)
    for n, c in [(1, "C0"), (5, "C1"), (30, "C2"), (200, "C3")]:
        means = clt_sample_means(sampler, n_per_sample=n, n_samples=20_000)  # (20000,)
        z = (means - 1.0) * np.sqrt(n)  # standardised: (mean - mu) / (sigma / sqrt n)
        ax.hist(z, bins=80, range=(-4, 4), density=True, histtype="step", lw=1.5, color=c, label=f"n = {n}")
    g = np.linspace(-4, 4, 300)
    ax.plot(g, np.exp(-0.5 * g**2) / np.sqrt(2 * np.pi), "k--", lw=1, label="N(0,1)")
    ax.set_title("CLT: standardised means of Exponential(1) draws")
    ax.set_xlabel("√n (x̄ − μ) / σ")
    ax.legend(fontsize=8)

    ax = axes[1]
    p0 = 0.05
    z_alpha = z_critical(0.95)
    for n, c in [(5_000, "C0"), (20_000, "C1"), (80_000, "C2")]:
        lifts = np.linspace(0, 0.012, 200)  # absolute lift in conversion rate
        p1 = p0 + lifts
        se = np.sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)  # (200,)
        power = 1 - normal_cdf(z_alpha - lifts / se) + normal_cdf(-z_alpha - lifts / se)
        ax.plot(lifts * 100, power, color=c, lw=2, label=f"{n:,} users / arm")
    ax.axhline(0.8, color="0.5", ls="--", lw=1)
    ax.text(0.02, 0.82, "80% power", fontsize=8, color="0.4")
    ax.set_xlabel("true absolute lift (percentage points) on a 5% baseline")
    ax.set_ylabel("power = P(reject H₀ | lift)")
    ax.set_title("Two-proportion z-test power (α = 0.05, two-sided)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
