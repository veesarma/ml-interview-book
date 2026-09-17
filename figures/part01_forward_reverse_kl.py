"""Forward KL(p||q) is mode-covering; reverse KL(q||p) is mode-seeking."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlbook.math.info_theory import fit_gaussian_to_mixture_kl

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part01_forward_reverse_kl.png"


def main() -> None:
    xs = np.linspace(-8, 8, 401)  # (G,)
    p = 0.6 * np.exp(-0.5 * ((xs + 3) / 0.8) ** 2) + 0.4 * np.exp(-0.5 * ((xs - 3) / 0.8) ** 2)
    p /= p.sum()
    fits = {d: fit_gaussian_to_mixture_kl(xs, p, d) for d in ("forward", "reverse")}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    titles = {
        "forward": "min KL(p ‖ q): q must cover every mode of p\n(MLE, distillation with teacher samples)",
        "reverse": "min KL(q ‖ p): q sits on one mode, never where p≈0\n(RLHF KL penalty, variational inference)",
    }
    for ax, d in zip(axes, ("forward", "reverse")):
        mu, sig = fits[d]
        q = np.exp(-0.5 * ((xs - mu) / sig) ** 2)
        q /= q.sum()
        ax.fill_between(xs, p, alpha=0.3, color="C0", label="target p (bimodal)")
        ax.plot(xs, q, color="C3", lw=2.5, label=f"best Gaussian q: μ={mu:.1f}, σ={sig:.1f}")
        ax.set_title(titles[d], fontsize=10)
        ax.set_xlabel("x")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("probability mass (grid)")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
