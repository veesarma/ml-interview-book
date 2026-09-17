"""Byte-pair encoding: training and encoding from scratch, word-level (Sennrich 2016) and
byte-level (GPT-2).

Training loop (identical for both):
    1. split the corpus into "words" (pre-tokenisation), count them
    2. represent each word as a tuple of symbols (characters or bytes)
    3. repeat until vocab_size: count adjacent symbol pairs across all words (weighted by
       word frequency), merge the most frequent pair into a new symbol, record the merge
Encoding a new word replays the recorded merges in the order they were learned.
"""

from __future__ import annotations

import re
from collections import Counter

Pair = tuple[str, str]


def count_pairs(word_freqs: dict[tuple[str, ...], int]) -> Counter:
    """Frequency of every adjacent symbol pair, weighted by how often the word occurs."""
    pairs: Counter = Counter()
    for symbols, freq in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge_pair(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    """Replace every non-overlapping occurrence of ``pair`` in ``symbols`` by the joined symbol."""
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def learn_merges(word_freqs: dict[tuple[str, ...], int], num_merges: int) -> list[Pair]:
    """Greedy BPE training: ``num_merges`` most-frequent-pair merges, in order."""
    merges: list[Pair] = []
    for _ in range(num_merges):
        pairs = count_pairs(word_freqs)
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]  # ties broken deterministically
        word_freqs = {merge_pair(w, best): f for w, f in word_freqs.items()}
        merges.append(best)
    return merges


def apply_merges(symbols: tuple[str, ...], ranks: dict[Pair, int]) -> tuple[str, ...]:
    """Encode one word: repeatedly merge the present pair with the lowest rank (earliest learned)."""
    while len(symbols) > 1:
        candidates = [(ranks[(a, b)], (a, b)) for a, b in zip(symbols, symbols[1:]) if (a, b) in ranks]
        if not candidates:
            break
        symbols = merge_pair(symbols, min(candidates)[1])
    return symbols


class BPETokenizer:
    """Word-level BPE with an end-of-word marker. ``train`` then ``encode``/``decode``."""

    def __init__(self, end_of_word: str = "</w>", unk_token: str = "<unk>") -> None:
        self.eow = end_of_word
        self.unk = unk_token
        self.merges: list[Pair] = []
        self.ranks: dict[Pair, int] = {}
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        return tuple(word[:-1]) + (word[-1] + self.eow,)  # last char carries the marker

    def train(self, corpus: str, vocab_size: int) -> None:
        words = Counter(corpus.split())
        word_freqs = {self._word_to_symbols(w): f for w, f in words.items()}
        chars = {c for w in words for c in w}
        alphabet = [self.unk] + sorted(chars) + sorted(c + self.eow for c in chars)  # every char, with and without the marker
        self.merges = learn_merges(word_freqs, max(0, vocab_size - len(alphabet)))
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        tokens = alphabet + [a + b for a, b in self.merges]
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def tokenize(self, text: str) -> list[str]:
        out: list[str] = []
        for w in text.split():
            out.extend(apply_merges(self._word_to_symbols(w), self.ranks))
        return out

    def encode(self, text: str) -> list[int]:
        unk = self.vocab[self.unk]
        return [self.vocab.get(t, unk) for t in self.tokenize(text)]  # characters never seen in training -> <unk>

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab[i] for i in ids).replace(self.eow, " ").strip()


# ---------------------------------------------------------------------------
# Byte-level BPE (GPT-2)
# ---------------------------------------------------------------------------


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible map from the 256 byte values to printable unicode characters,
    so that merges can be stored as text and no byte is ever "unknown"."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# GPT-2's pattern uses \p{L}/\p{N} (needs the `regex` module); this ASCII-letter approximation
# keeps the same structure: contractions, space-prefixed words, numbers, punctuation runs, spaces.
PRETOKENIZE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class ByteLevelBPE:
    """GPT-2 style: pre-tokenise with a regex, map UTF-8 bytes to unicode symbols, BPE over those.
    Base vocabulary is exactly 256 symbols, so any string encodes (no [UNK])."""

    def __init__(self) -> None:
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {c: b for b, c in self.byte_encoder.items()}
        self.merges: list[Pair] = []
        self.ranks: dict[Pair, int] = {}
        self.vocab: dict[str, int] = {}
        self.inv_vocab: dict[int, str] = {}

    def _pretokens(self, text: str) -> list[tuple[str, ...]]:
        chunks = PRETOKENIZE.findall(text)
        return [tuple(self.byte_encoder[b] for b in chunk.encode("utf-8")) for chunk in chunks]

    def train(self, corpus: str, vocab_size: int) -> None:
        word_freqs = Counter(self._pretokens(corpus))
        base = [self.byte_encoder[b] for b in range(256)]
        self.merges = learn_merges(dict(word_freqs), max(0, vocab_size - 256))
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        tokens = base + [a + b for a, b in self.merges]
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.inv_vocab = {i: t for t, i in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for symbols in self._pretokens(text):
            ids.extend(self.vocab[t] for t in apply_merges(symbols, self.ranks))
        return ids

    def decode(self, ids: list[int]) -> str:
        text = "".join(self.inv_vocab[i] for i in ids)
        return bytes(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")
