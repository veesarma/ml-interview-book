# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/tokenizer_bpe.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tokenizer_bpe -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/tokenizer_bpe --force

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
    raise NotImplementedError('TODO: implement count_pairs (see the reference in src/mlbook)')

def merge_pair(symbols: tuple[str, ...], pair: Pair) -> tuple[str, ...]:
    """Replace every non-overlapping occurrence of ``pair`` in ``symbols`` by the joined symbol."""
    raise NotImplementedError('TODO: implement merge_pair (see the reference in src/mlbook)')

def learn_merges(word_freqs: dict[tuple[str, ...], int], num_merges: int) -> list[Pair]:
    """Greedy BPE training: ``num_merges`` most-frequent-pair merges, in order."""
    raise NotImplementedError('TODO: implement learn_merges (see the reference in src/mlbook)')

def apply_merges(symbols: tuple[str, ...], ranks: dict[Pair, int]) -> tuple[str, ...]:
    """Encode one word: repeatedly merge the present pair with the lowest rank (earliest learned)."""
    raise NotImplementedError('TODO: implement apply_merges (see the reference in src/mlbook)')

class BPETokenizer:
    """Word-level BPE with an end-of-word marker. ``train`` then ``encode``/``decode``."""

    def __init__(self, end_of_word: str='</w>', unk_token: str='<unk>') -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        raise NotImplementedError('TODO: implement _word_to_symbols (see the reference in src/mlbook)')

    def train(self, corpus: str, vocab_size: int) -> None:
        raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError('TODO: implement tokenize (see the reference in src/mlbook)')

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible map from the 256 byte values to printable unicode characters,
    so that merges can be stored as text and no byte is ever "unknown"."""
    raise NotImplementedError('TODO: implement bytes_to_unicode (see the reference in src/mlbook)')
PRETOKENIZE = re.compile("'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+(?!\\S)|\\s+")

class ByteLevelBPE:
    """GPT-2 style: pre-tokenise with a regex, map UTF-8 bytes to unicode symbols, BPE over those.
    Base vocabulary is exactly 256 symbols, so any string encodes (no [UNK])."""

    def __init__(self) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _pretokens(self, text: str) -> list[tuple[str, ...]]:
        raise NotImplementedError('TODO: implement _pretokens (see the reference in src/mlbook)')

    def train(self, corpus: str, vocab_size: int) -> None:
        raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')
