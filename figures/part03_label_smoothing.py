"""Label smoothing: the CE loss of a 2-class logit gap with and without smoothing,
and the resulting optimum. -> docs/assets/figures/part03_label_smoothing.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

gap = np.linspace(-2, 12, 500)  # z_true - z_other for K = 2
K = 2
fig, ax = plt.subplots(figsize=(6.5, 3.6), facecolor="white")
for eps in [0.0, 0.05, 0.1, 0.2]:
    q_true = 1 - eps + eps / K
    logp_true = -np.log1p(np.exp(-gap))
    logp_other = -np.log1p(np.exp(gap))
    loss = -(q_true * logp_true + (1 - q_true) * logp_other)
    ax.plot(gap, loss, label=f"eps = {eps}", lw=1.7)
    if eps > 0:
        opt = np.log((1 - eps + eps / K) / (eps / K))
        ax.axvline(opt, color="gray", lw=0.6, ls=":")
        ax.text(opt, 1.05, f"{opt:.1f}", fontsize=7, ha="center")
ax.set_xlabel("logit gap  z_true - z_other")
ax.set_ylabel("cross-entropy with smoothed target")
ax.set_ylim(0, 1.2)
ax.set_title("Hard targets push the gap to infinity; smoothing gives a finite optimum log((K-1)(1-eps)/eps + 1)", fontsize=8.5)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("docs/assets/figures/part03_label_smoothing.png", dpi=150, bbox_inches="tight", facecolor="white")
