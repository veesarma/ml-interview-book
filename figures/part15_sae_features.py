"""Sparse autoencoder recovering superposed features: cosine similarity matrix and sparsity."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mlbook.interp.sparse_autoencoder import SparseAutoencoder, make_superposition_data, train_sae

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part15_sae_features.png"


def main() -> None:
    torch.manual_seed(0)
    torch.set_num_threads(1)
    X, dirs = make_superposition_data(n=4096, d=16, n_true=32, p_active=0.05)
    sae = SparseAutoencoder(16, 64)
    train_sae(sae, X, l1_coeff=0.2, steps=600)
    cos = (dirs @ sae.W_dec.detach().T).abs()  # (32, 64)
    order = cos.argmax(dim=1)  # best feature for each true direction
    used = list(dict.fromkeys(order.tolist()))
    rest = [j for j in range(64) if j not in used]
    cos_sorted = cos[:, used + rest]
    _, f = sae(X[:512])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    im = axes[0].imshow(cos_sorted, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xlabel("learned SAE feature (columns sorted by best match)"); axes[0].set_ylabel("true superposed direction")
    axes[0].set_title("|cos| between true directions and decoder rows (32 true features in d=16)", fontsize=9)
    fig.colorbar(im, ax=axes[0])
    active = (f > 1e-3).float().sum(dim=1)
    axes[1].hist(active.numpy(), bins=range(0, 12), color="C0", edgecolor="white")
    axes[1].set_xlabel("active features per activation vector"); axes[1].set_ylabel("count")
    axes[1].set_title("L1 penalty -> sparse codes (data has ~1.6 active on average)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
