"""Feature store: a point-in-time-correct join versus the leakage of a naive latest-value join."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part17_point_in_time.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    # Feature value timeline for one entity: user_7d_txn_count
    feature_times = [1, 3, 5, 8, 11]
    feature_vals = [2, 3, 5, 9, 4]
    ax.step(feature_times + [14], feature_vals + [4], where="post", color="C0", lw=2, label="feature value as materialised (user_7d_txn_count)")
    for t, v in zip(feature_times, feature_vals):
        ax.scatter([t], [v], color="C0", zorder=3, s=30)
    # Label events (training rows) at times 4, 7, 10
    label_times = [4, 7, 10]
    for lt in label_times:
        ax.axvline(lt, color="C3", ls=":", lw=1.5)
        ax.text(lt + 0.1, 9.6, f"label\nevent\nt={lt}", fontsize=8, color="C3")
        # correct: last value strictly before lt
        correct_val = [v for t, v in zip(feature_times, feature_vals) if t <= lt][-1]
        ax.scatter([lt], [correct_val], marker="s", color="C2", s=70, zorder=4,
                   label="point-in-time join: value known at label time" if lt == 4 else None)
        ax.scatter([lt], [feature_vals[-1]], marker="x", color="C1", s=80, zorder=4,
                   label="naive 'latest' join: value from the future (leakage)" if lt == 4 else None)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("feature value")
    ax.set_ylim(0, 11)
    ax.set_xlim(0, 14)
    ax.set_title("Point-in-time correctness: each training row sees only what the online model could have seen")
    ax.legend(fontsize=8, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
