"""BadNets-style trigger on a tiny classifier and the spectral-signature defence."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mlbook.safety.backdoor_demo import (TinyClassifier, accuracy, attack_success_rate, make_clean_data, poison,
                                         spectral_signature_scores, stamp_trigger, train)

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part15_backdoor.png"


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    X, y = make_clean_data(600)
    Xp, yp, mask = poison(X, y, frac=0.15)
    model = TinyClassifier()
    train(model, Xp, yp, epochs=250)
    Xt, yt = make_clean_data(300, seed=1)
    acc, asr = accuracy(model, Xt, yt), attack_success_rate(model, Xt, yt)
    scores, idx = spectral_signature_scores(model, Xp, yp)
    i0 = int(torch.nonzero(y == 0)[0])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].imshow(X[i0, 0], cmap="gray", vmin=0, vmax=1); axes[0].set_title(f"clean, label 0\nmodel says {int(model(X[i0:i0+1]).argmax())}", fontsize=9)
    xt = stamp_trigger(X[i0:i0 + 1])
    axes[1].imshow(xt[0, 0], cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"+ trigger (2x2 patch)\nmodel says {int(model(xt).argmax())}", fontsize=9)
    for ax in axes[:2]:
        ax.axis("off")
    axes[2].hist(scores[~mask[idx]].numpy(), bins=25, alpha=0.7, label="clean class-1 examples")
    axes[2].hist(scores[mask[idx]].numpy(), bins=25, alpha=0.7, label="poisoned (relabelled)")
    axes[2].set_xlabel("|projection on top singular vector| of centred features"); axes[2].legend(fontsize=8)
    axes[2].set_title(f"spectral signature; clean acc {acc:.2f}, attack success {asr:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
