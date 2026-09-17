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

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer. 'Hello, World' -> ['hello', 'world']."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 index over a list of documents.

    fit(docs) builds term frequencies, document lengths and IDF.
    score(query) returns a (N,) array of scores; topk(query, k) returns ids and scores.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tfs: list[Counter] = []
        self.doc_lens: np.ndarray = np.zeros(0)  # (N,)
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.n_docs: int = 0

    def fit(self, docs: list[str]) -> "BM25":
        """Index the corpus. docs: list of N strings."""
        self.doc_tfs = [Counter(tokenize(d)) for d in docs]
        self.doc_lens = np.array([sum(tf.values()) for tf in self.doc_tfs], dtype=float)  # (N,)
        self.n_docs = len(docs)
        self.avgdl = float(self.doc_lens.mean()) if self.n_docs else 0.0
        df: Counter = Counter()
        for tf in self.doc_tfs:
            df.update(tf.keys())
        # IDF with the "+1" inside the log so it is never negative (Lucene convention).
        self.idf = {
            t: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()
        }
        return self

    def term_score(self, tf: float, doc_len: float) -> float:
        """Saturated, length-normalised term frequency (the BM25 'TF' factor)."""
        denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
        return tf * (self.k1 + 1.0) / denom

    def score(self, query: str) -> np.ndarray:
        """BM25 score of every document for the query. -> (N,)."""
        scores = np.zeros(self.n_docs)  # (N,)
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in enumerate(self.doc_tfs):
                f = tf.get(term, 0)
                if f:
                    scores[i] += idf * self.term_score(f, self.doc_lens[i])
        return scores

    def topk(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Top-k document ids and scores, best first. -> ((k,), (k,))."""
        s = self.score(query)  # (N,)
        k = min(k, self.n_docs)
        ids = np.argsort(-s, kind="stable")[:k]  # (k,)
        return ids, s[ids]
