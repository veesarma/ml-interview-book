"""AUC improves while the shipped operating point gets worse.

Two synthetic fraud-style scorers on the same 2% positive-rate population.
Model B orders the bulk of the population better (higher ROC-AUC) but pushes a
small slice of negatives into the very top of the score list, which is exactly
where the review queue operates. Panel 1: ROC curves with the shipped operating
point. Panel 2: precision as a function of the review-queue size, with the
budget marked. The script prints the numbers quoted in the chapter.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREY = "#1f77b4", "#ff7f0e", "#7f7f7f"

N = 200_000
PREVALENCE = 0.02
QUEUE = 2_000  # analysts can review this many alerts per day


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5",
        "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    })


def roc(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (fpr, tpr, auc). y in {0,1} of shape (N,), s of shape (N,)."""
    order = np.argsort(-s)  # (N,)
    y_sorted = y[order]  # (N,)
    tps = np.cumsum(y_sorted)  # (N,)
    fps = np.cumsum(1 - y_sorted)  # (N,)
    tpr = np.concatenate([[0.0], tps / tps[-1]])  # (N+1,)
    fpr = np.concatenate([[0.0], fps / fps[-1]])  # (N+1,)
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def precision_at_k(y: np.ndarray, s: np.ndarray, ks: np.ndarray) -> np.ndarray:
    order = np.argsort(-s)  # (N,)
    hits = np.cumsum(y[order])  # (N,)
    return hits[ks - 1] / ks  # (len(ks),)


def main() -> None:
    style()
    rng = np.random.default_rng(7)
    y = (rng.random(N) < PREVALENCE).astype(int)  # (N,)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    # Model A: moderate global separation, clean top of the list.
    a = np.empty(N)  # (N,)
    a[y == 1] = rng.normal(1.6, 1.0, n_pos)
    a[y == 0] = rng.normal(0.0, 1.0, n_neg)

    # Model B: better separation over the whole population, but a small group of
    # negatives (a new feature that fires on a benign merchant category) is
    # scored very high and floods the top of the queue.
    b = np.empty(N)  # (N,)
    b[y == 1] = rng.normal(2.1, 1.0, n_pos)
    b[y == 0] = rng.normal(0.0, 1.0, n_neg)
    neg_idx = np.flatnonzero(y == 0)  # (n_neg,)
    poisoned = rng.choice(neg_idx, size=int(0.008 * n_neg), replace=False)  # (0.008*n_neg,)
    b[poisoned] = rng.normal(4.4, 0.5, poisoned.size)

    fpr_a, tpr_a, auc_a = roc(y, a)
    fpr_b, tpr_b, auc_b = roc(y, b)
    ks = np.arange(200, 20_001, 100)  # (198,)
    prec_a, prec_b = precision_at_k(y, a, ks), precision_at_k(y, b, ks)
    pa = float(precision_at_k(y, a, np.array([QUEUE]))[0])
    pb = float(precision_at_k(y, b, np.array([QUEUE]))[0])

    # operating point of the queue: the threshold that admits QUEUE alerts
    thr_a = np.sort(a)[-QUEUE]
    thr_b = np.sort(b)[-QUEUE]
    op_a = ((a >= thr_a) & (y == 0)).sum() / n_neg, ((a >= thr_a) & (y == 1)).sum() / n_pos
    op_b = ((b >= thr_b) & (y == 0)).sum() / n_neg, ((b >= thr_b) & (y == 1)).sum() / n_pos

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(fpr_a, tpr_a, color=BLUE, lw=1.8, label=f"model A, AUC = {auc_a:.3f}")
    ax.plot(fpr_b, tpr_b, color=ORANGE, lw=1.8, label=f"model B, AUC = {auc_b:.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls=":")
    ax.scatter(*op_a, color=BLUE, zorder=5, s=45, edgecolor="white")
    ax.scatter(*op_b, color=ORANGE, zorder=5, s=45, edgecolor="white")
    ax.annotate("shipped operating point\n(top 2,000 alerts / day)", xy=op_a,
                xytext=(0.30, 0.42), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate (recall)")
    ax.set_title("B wins on AUC over the whole curve")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(ks, prec_a, color=BLUE, lw=1.8, label="model A")
    ax.plot(ks, prec_b, color=ORANGE, lw=1.8, label="model B")
    ax.axvline(QUEUE, color="black", lw=1.0, ls="--")
    ax.annotate(f"review budget\nprecision {pa:.2f} vs {pb:.2f}", xy=(QUEUE, max(pa, pb)),
                xytext=(4200, 0.78 * max(prec_a.max(), prec_b.max())), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.set_xlabel("alerts reviewed per day (top-$k$ by score)")
    ax.set_ylabel("precision@$k$")
    ax.set_title("A wins where the business actually operates")
    ax.legend(loc="upper right")

    fig.tight_layout()
    path = OUT / "part19_auc_kpi.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print("wrote", path)
    print(f"AUC A={auc_a:.3f} B={auc_b:.3f}")
    print(f"precision@{QUEUE}: A={pa:.3f} B={pb:.3f}")
    print(f"recall@{QUEUE}: A={op_a[1]:.3f} B={op_b[1]:.3f}")


if __name__ == "__main__":
    main()
