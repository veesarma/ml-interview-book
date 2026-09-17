# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/transformer/tokenizer_wordpiece.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k tokenizer_wordpiece -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py transformer/tokenizer_wordpiece --force

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

    def __init__(self, unk_token: str='[UNK]', prefix: str='##') -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _word_to_symbols(self, word: str) -> tuple[str, ...]:
        raise NotImplementedError('TODO: implement _word_to_symbols (see the reference in src/mlbook)')

    def _join(self, a: str, b: str) -> str:
        raise NotImplementedError('TODO: implement _join (see the reference in src/mlbook)')

    def train(self, corpus: str, vocab_size: int) -> None:
        raise NotImplementedError('TODO: implement train (see the reference in src/mlbook)')

    def tokenize_word(self, word: str) -> list[str]:
        """Greedy longest-match-first from the left; any failure makes the whole word [UNK]."""
        raise NotImplementedError('TODO: implement tokenize_word (see the reference in src/mlbook)')

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError('TODO: implement tokenize (see the reference in src/mlbook)')

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError('TODO: implement encode (see the reference in src/mlbook)')

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError('TODO: implement decode (see the reference in src/mlbook)')

def viterbi_segment(word: str, log_probs: dict[str, float]) -> tuple[list[str], float]:
    """Most probable segmentation of ``word`` into pieces from ``log_probs``.

    best[i] = max_{j<i, word[j:i] in vocab} best[j] + log p(word[j:i]).  Returns (pieces, log-prob).
    """
    raise NotImplementedError('TODO: implement viterbi_segment (see the reference in src/mlbook)')

def unigram_em_step(word_freqs: dict[str, int], log_probs: dict[str, float]) -> dict[str, float]:
    """One hard-EM step: re-estimate piece probabilities from Viterbi segmentations.

    (SentencePiece uses the forward-backward soft counts; hard EM keeps the sketch short.)
    """
    raise NotImplementedError('TODO: implement unigram_em_step (see the reference in src/mlbook)')

def unigram_prune(word_freqs: dict[str, int], log_probs: dict[str, float], keep_fraction: float=0.8) -> dict[str, float]:
    """Drop the pieces whose removal increases the corpus negative log-likelihood the least,
    never dropping single characters (they guarantee every word stays segmentable)."""
    raise NotImplementedError('TODO: implement unigram_prune (see the reference in src/mlbook)')
