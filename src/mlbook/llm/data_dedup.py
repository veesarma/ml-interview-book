"""Deduplication and decontamination utilities for pretraining corpora (pure NumPy).

Three tools every LLM data pipeline needs:

* ``exact_dedup``          – drop byte-identical documents via a content hash.
* ``MinHash`` + ``LSHIndex`` – *near*-duplicate detection.  A MinHash signature of
  ``num_perm`` hashes estimates the Jaccard similarity of two shingle sets:
  ``P[min_h(A) == min_h(B)] = |A ∩ B| / |A ∪ B|``.  Locality-sensitive hashing
  (banding) turns pairwise comparison into a candidate lookup.
* ``ngram_contamination``  – the GPT-3 / Llama style "does any evaluation n-gram
  appear in the training set" check.

Everything is documented with shapes; signatures are ``(num_perm,)`` uint64 arrays.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

_MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1, a prime > any 32-bit hash value
_MAX_HASH = (1 << 32) - 1  # shingle hashes are reduced to 32 bits

_TOKEN_RE = re.compile(r"\w+")


def normalise(text: str) -> list[str]:
    """Lower-case, strip punctuation, and split into word tokens.

    Args:
        text: raw document string.
    Returns:
        list of ``n_words`` lower-cased word tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def shingles(text: str, n: int = 5) -> set[str]:
    """Word n-gram (shingle) set of a document.

    A document with ``n_words`` tokens has ``max(0, n_words - n + 1)`` shingles.
    Documents shorter than ``n`` words get their whole text as one shingle so
    they are never silently empty.
    """
    words = normalise(text)
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _hash32(s: str) -> int:
    """Stable 32-bit hash of a string (first 4 bytes of SHA-1)."""
    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:4], "little")


def exact_dedup(docs: list[str]) -> list[int]:
    """Indices of the first occurrence of each distinct document (byte-exact).

    Args:
        docs: list of ``n_docs`` strings.
    Returns:
        sorted list of kept indices (one per distinct SHA-256 hash).
    """
    seen: set[str] = set()
    keep: list[int] = []
    for i, d in enumerate(docs):
        h = hashlib.sha256(d.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            keep.append(i)
    return keep


def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity ``|a ∩ b| / |a ∪ b|`` (0.0 when both empty)."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class MinHash:
    """MinHash signatures with ``num_perm`` universal hash functions.

    Each hash is ``h_i(x) = ((a_i * x + b_i) mod p) mod 2^32`` with ``p`` a
    Mersenne prime.  The signature of a set is the element-wise minimum over
    the set's shingle hashes, shape ``(num_perm,)``.
    """

    num_perm: int = 128
    seed: int = 0
    a: np.ndarray = field(init=False, repr=False)  # (num_perm,) multipliers
    b: np.ndarray = field(init=False, repr=False)  # (num_perm,) offsets

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # a, b < 2^31 and x < 2^32 keep a*x + b < 2^64, so uint64 arithmetic never overflows.
        self.a = rng.integers(1, 1 << 31, size=self.num_perm, dtype=np.uint64)  # (num_perm,)
        self.b = rng.integers(0, 1 << 31, size=self.num_perm, dtype=np.uint64)  # (num_perm,)

    def signature(self, shingle_set: set[str]) -> np.ndarray:
        """MinHash signature of one document.

        Args:
            shingle_set: set of ``n_shingles`` strings.
        Returns:
            (num_perm,) uint64 array of per-permutation minimum hashes.
        """
        if not shingle_set:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)  # (num_perm,)
        x = np.array([_hash32(s) for s in shingle_set], dtype=np.uint64)  # (n_shingles,)
        affine = self.a[None, :] * x[:, None] + self.b[None, :]  # (n_shingles, num_perm)
        hashes = (affine % np.uint64(_MERSENNE_PRIME)) & np.uint64(_MAX_HASH)  # (n_shingles, num_perm)
        return hashes.min(axis=0)  # (num_perm,)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of agreeing signature slots, an unbiased estimate of Jaccard.

    Args:
        sig_a, sig_b: (num_perm,) signatures.
    """
    return float(np.mean(sig_a == sig_b))


class LSHIndex:
    """Banded locality-sensitive hashing over MinHash signatures.

    A signature of ``num_perm`` hashes is cut into ``bands`` bands of ``rows``
    hashes each (``num_perm = bands * rows``).  Two documents are candidates if
    *any* band matches exactly.  With Jaccard ``s`` the candidate probability is
    ``1 - (1 - s^rows)^bands``, an S-curve with threshold ~``(1/bands)^(1/rows)``.
    """

    def __init__(self, num_perm: int, bands: int) -> None:
        if num_perm % bands != 0:
            raise ValueError("num_perm must be divisible by bands")
        self.bands = bands
        self.rows = num_perm // bands
        self.tables: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        sig = sig.reshape(self.bands, self.rows)  # (bands, rows)
        return [sig[b].tobytes() for b in range(self.bands)]

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        """Insert a (num_perm,) signature under ``doc_id``."""
        for table, key in zip(self.tables, self._band_keys(sig)):
            table[key].append(doc_id)

    def query(self, sig: np.ndarray) -> set[int]:
        """All doc ids sharing at least one band with ``sig`` (candidate pairs)."""
        out: set[int] = set()
        for table, key in zip(self.tables, self._band_keys(sig)):
            out.update(table.get(key, ()))
        return out

    def threshold(self) -> float:
        """Jaccard value at which the S-curve crosses 0.5, ``(1/bands)^(1/rows)``."""
        return (1.0 / self.bands) ** (1.0 / self.rows)


def near_dedup(
    docs: list[str], n: int = 5, num_perm: int = 128, bands: int = 16, threshold: float = 0.8
) -> list[int]:
    """Keep one representative per near-duplicate cluster.

    Pipeline: shingle → MinHash → LSH candidates → verify estimated Jaccard ≥ threshold.
    Args:
        docs: ``n_docs`` strings.
    Returns:
        sorted kept indices (first document of each cluster wins).
    """
    mh = MinHash(num_perm=num_perm)
    index = LSHIndex(num_perm=num_perm, bands=bands)
    sigs: list[np.ndarray] = []  # each (num_perm,)
    keep: list[int] = []
    for i, d in enumerate(docs):
        sig = mh.signature(shingles(d, n))  # (num_perm,)
        duplicate = any(estimate_jaccard(sig, sigs[j]) >= threshold for j in index.query(sig))
        sigs.append(sig)
        if not duplicate:
            keep.append(i)
            index.add(i, sig)
    return keep


def ngram_contamination(train_docs: list[str], eval_docs: list[str], n: int = 13) -> dict:
    """GPT-3 / Llama style n-gram overlap contamination check.

    An evaluation document is *contaminated* if any of its word n-grams occurs
    anywhere in the training corpus.  Returns the fraction of contaminated eval
    documents and their indices.  ``n=13`` was used by GPT-3; Llama 3 uses
    token-level 8-grams with a per-benchmark threshold on the overlap ratio.
    """
    train_ngrams: set[str] = set()
    for d in train_docs:
        train_ngrams |= shingles(d, n)
    flagged: list[int] = []
    for i, d in enumerate(eval_docs):
        if shingles(d, n) & train_ngrams:
            flagged.append(i)
    frac = len(flagged) / max(1, len(eval_docs))
    return {"contaminated_fraction": frac, "contaminated_indices": flagged}
