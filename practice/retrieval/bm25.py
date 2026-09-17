# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/bm25.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k bm25 -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/bm25 --force

"""BM25 (Okapi) sparse retrieval from scratch (pure Python + NumPy).

Score of document D for query terms q_1..q_n:

    score(q, D) = sum_i IDF(q_i) * f(q_i, D) (k1 + 1)
                                  / ( f(q_i, D) + k1 (1 - b + b |D| / avgdl) )

    IDF(t) = log( (N - n_t + 0.5) / (n_t + 0.5) + 1 )

where f(t, D) is the term frequency in D, |D| the document length in tokens,
avgdl the mean document length, N the corpus size and n_t the number of documents
containing t. k1 controls term-frequency saturation, b controls length normalisation.
"""
from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np
_TOKEN_RE = re.compile('[a-z0-9]+')

def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer. 'Hello, World' -> ['hello', 'world']."""
    raise NotImplementedError('TODO: implement tokenize (see the reference in src/mlbook)')

class BM25:
    """Okapi BM25 index over a list of documents.

    fit(docs) builds term frequencies, document lengths and IDF.
    score(query) returns a (N,) array of scores; topk(query, k) returns ids and scores.
    """

    def __init__(self, k1: float=1.5, b: float=0.75) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def fit(self, docs: list[str]) -> 'BM25':
        """Index the corpus. docs: list of N strings."""
        raise NotImplementedError('TODO: implement fit (see the reference in src/mlbook)')

    def term_score(self, tf: float, doc_len: float) -> float:
        """Saturated, length-normalised term frequency (the BM25 'TF' factor)."""
        raise NotImplementedError('TODO: implement term_score (see the reference in src/mlbook)')

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every document for the query. -> (N,)."""
        raise NotImplementedError('TODO: implement score (see the reference in src/mlbook)')

    def topk(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k document ids and scores, best first. -> ((k,), (k,))."""
        raise NotImplementedError('TODO: implement topk (see the reference in src/mlbook)')
