"""ETA forecasting: the pinball (quantile) loss and a right-skewed delivery-time distribution with its quantiles."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_quantile_eta.png"


def pinball(u: np.ndarray, tau: float) -> np.ndarray:
    return np.maximum(tau * u, (tau - 1) * u)


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    u = np.linspace(-10, 10, 400)
    for tau, c in [(0.1, "C0"), (0.5, "C1"), (0.9, "C2")]:
        ax1.plot(u, pinball(u, tau), lw=2, color=c, label=f"tau = {tau}")
    ax1.set_xlabel("residual u = y - q  (minutes)")
    ax1.set_ylabel("pinball loss")
    ax1.set_title("Quantile loss: asymmetric penalty picks the quantile")
    ax1.legend(fontsize=8)

    rng = np.random.default_rng(5)
    eta = rng.lognormal(mean=np.log(28), sigma=0.35, size=100_000)
    ax2.hist(eta, bins=120, range=(0, 90), color="C0", alpha=0.6, density=True)
    for q, c, lab in [(0.5, "C1", "median (what you show)"), (0.9, "C2", "p90 (what you promise)"), (0.99, "C3", "p99 (what ops plan for)")]:
        v = np.quantile(eta, q)
        ax2.axvline(v, color=c, lw=2, label=f"{lab}: {v:.0f} min")
    ax2.set_xlabel("delivery time (minutes)")
    ax2.set_ylabel("density")
    ax2.set_title("Right-skewed: the mean is a bad promise")
    ax2.legend(fontsize=8)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
