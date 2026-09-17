"""WordPiece (BERT) training + greedy longest-match encoding, and a Unigram-LM sketch.

WordPiece differs from BPE only in the merge criterion:
    BPE:       merge the pair with the highest count(ab)
    WordPiece: merge the pair with the highest count(ab) / (count(a) * count(b))
i.e. the pair whose merge most increases the likelihood of the corpus under a unigram
model over the current vocabulary. Continuation pieces carry the "##" prefix.

Unigram LM (Kudo 2018; SentencePiece's default) goes the other way: start from a large
candidate vocabulary, fit piece probabilities with EM, and prune pieces whose removal
hurts the corpus likelihood least. Segmentation is Viterbi over the lattice of pieces.
"""

from __future__ import annotations

import math
from collections import Counter

from .tokenizer_bpe import Pair, merge_pair


class WordPieceTokenizer:
    def __init__(self, unk_token: str = "[UNK]", prefix: str = "##") -> None:
        self.unk, self.prefix = unk_token, prefix
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        return (word[0],) + tuple(self.prefix + c for c in word[1:])

    def _join(self, a: str, b: str) -> str:
        return a + b[len(self.prefix) :] if b.startswith(self.prefix) else a + b

    def train(self, corpus: str, vocab_size: int) -> None:
        word_freqs = {self._word_to_symbols(w): f for w, f in Counter(corpus.split()).items()}
        vocab = [self.unk] + sorted({s for w in word_freqs for s in w})
        while len(vocab) < vocab_size:
            pair_counts: Counter = Counter()
            sym_counts: Counter = Counter()
            for symbols, freq in word_freqs.items():
                for s in symbols:
                    sym_counts[s] += freq
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            # likelihood-based score: count(ab) / (count(a) count(b))
            best: Pair = max(pair_counts, key=lambda p: (pair_counts[p] / (sym_counts[p[0]] * sym_counts[p[1]]), p))
            new_sym = self._join(*best)
            word_freqs = {tuple(new_sym if s == best[0] + best[1] else s for s in merge_pair(w, best)): f for w, f in word_freqs.items()}
            vocab.append(new_sym)
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def tokenize_word(self, word: str) -> list[str]:
        """Greedy longest-match-first from the left; any failure makes the whole word [UNK]."""
        pieces: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            piece = None
            while start < end:
                cand = word[start:end] if start == 0 else self.prefix + word[start:end]
                if cand in self.vocab:
                    piece = cand
                    break
                end -= 1
            if piece is None:
                return [self.unk]
            pieces.append(piece)
            start = end
        return pieces

    def tokenize(self, text: str) -> list[str]:
        return [p for w in text.split() for p in self.tokenize_word(w)]

    def encode(self, text: str) -> list[int]:
        return [self.vocab[t] for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        out = ""
        for i in ids:
            t = self.inv_vocab[i]
            out += t[len(self.prefix) :] if t.startswith(self.prefix) else (" " if out else "") + t
        return out


# ---------------------------------------------------------------------------
# Unigram LM sketch
# ---------------------------------------------------------------------------


def viterbi_segment(word: str, log_probs: dict[str, float]) -> tuple[list[str], float]:
    """Most probable segmentation of ``word`` into pieces from ``log_probs``.

    best[i] = max_{j<i, word[j:i] in vocab} best[j] + log p(word[j:i]).  Returns (pieces, log-prob).
    """
    n = len(word)
    best = [-math.inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for i in range(1, n + 1):
        for j in range(i):
            piece = word[j:i]
            if piece in log_probs and best[j] + log_probs[piece] > best[i]:
                best[i] = best[j] + log_probs[piece]
                back[i] = j
    if best[n] == -math.inf:
        return [], -math.inf
    pieces: list[str] = []
    i = n
    while i > 0:
        pieces.append(word[back[i] : i])
        i = back[i]
    return pieces[::-1], best[n]


def unigram_em_step(word_freqs: dict[str, int], log_probs: dict[str, float]) -> dict[str, float]:
    """One hard-EM step: re-estimate piece probabilities from Viterbi segmentations.

    (SentencePiece uses the forward-backward soft counts; hard EM keeps the sketch short.)
    """
    counts: Counter = Counter()
    for word, freq in word_freqs.items():
        pieces, _ = viterbi_segment(word, log_probs)
        for p in pieces:
            counts[p] += freq
    total = sum(counts.values())
    return {p: math.log(counts[p] / total) for p in log_probs if counts[p] > 0}


def unigram_prune(word_freqs: dict[str, int], log_probs: dict[str, float], keep_fraction: float = 0.8) -> dict[str, float]:
    """Drop the pieces whose removal increases the corpus negative log-likelihood the least,
    never dropping single characters (they guarantee every word stays segmentable)."""
    base = sum(f * viterbi_segment(w, log_probs)[1] for w, f in word_freqs.items())
    losses: dict[str, float] = {}
    for piece in log_probs:
        if len(piece) == 1:
            continue
        without = {p: lp for p, lp in log_probs.items() if p != piece}
        losses[piece] = base - sum(f * viterbi_segment(w, without)[1] for w, f in word_freqs.items())
    n_keep = int(len(losses) * keep_fraction)
    keep = set(sorted(losses, key=losses.get, reverse=True)[:n_keep]) | {p for p in log_probs if len(p) == 1}
    return {p: lp for p, lp in log_probs.items() if p in keep}
