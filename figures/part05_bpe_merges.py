"""BPE on a small English corpus: corpus length in tokens vs. number of merges (the compression
curve), with the first merges annotated. Writes docs/assets/figures/part05_bpe_merges.png.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlbook.transformer.tokenizer_bpe import count_pairs, merge_pair  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part05_bpe_merges.png"

CORPUS = (
    "the transformer is a model that uses attention the attention mechanism lets every token "
    "attend to every other token the model is trained on tokens and the tokens come from a tokenizer "
    "the tokenizer learns merges from the corpus and the merges are applied in order the lower the "
    "number of tokens the cheaper the inference the newest models use byte level tokenizers "
    "attention attention attention the the the model model tokens tokens merges merges"
)


def main() -> None:
    words = Counter(CORPUS.split())
    word_freqs = {tuple(w[:-1]) + (w[-1] + "</w>",): f for w, f in words.items()}
    n_tokens = [sum(len(w) * f for w, f in word_freqs.items())]
    merges = []
    for _ in range(60):
        pairs = count_pairs(word_freqs)
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append((best, pairs[best]))
        word_freqs = {merge_pair(w, best): f for w, f in word_freqs.items()}
        n_tokens.append(sum(len(w) * f for w, f in word_freqs.items()))
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor="white")
    ax.plot(range(len(n_tokens)), n_tokens, marker=".", color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0])
    for i in (0, 1, 2, 5, 10, 20, 40):
        if i < len(merges):
            (a, b), c = merges[i]
            ax.annotate(f"#{i + 1}: '{a}'+'{b}' ({c}x)", (i + 1, n_tokens[i + 1]), textcoords="offset points", xytext=(8, 10), fontsize=7.5, arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
    ax.set_xlabel("number of merges learned (vocabulary size - alphabet size)")
    ax.set_ylabel("corpus length in tokens")
    ax.set_title("BPE compression curve on a 100-word toy corpus", loc="left", fontsize=10)
    ax.grid(alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
