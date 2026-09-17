# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/llm/data_dedup.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k data_dedup -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py llm/data_dedup --force

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
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1
_TOKEN_RE = re.compile('\\w+')

def normalise(text: str) -> list[str]:
    """Lower-case, strip punctuation, and split into word tokens.

    Args:
        text: raw document string.
    Returns:
        list of ``n_words`` lower-cased word tokens.
    """
    raise NotImplementedError('TODO: implement normalise (see the reference in src/mlbook)')

def shingles(text: str, n: int=5) -> set[str]:
    """Word n-gram (shingle) set of a document.

    A document with ``n_words`` tokens has ``max(0, n_words - n + 1)`` shingles.
    Documents shorter than ``n`` words get their whole text as one shingle so
    they are never silently empty.
    """
    raise NotImplementedError('TODO: implement shingles (see the reference in src/mlbook)')

def _hash32(s: str) -> int:
    """Stable 32-bit hash of a string (first 4 bytes of SHA-1)."""
    raise NotImplementedError('TODO: implement _hash32 (see the reference in src/mlbook)')

def exact_dedup(docs: list[str]) -> list[int]:
    """Indices of the first occurrence of each distinct document (byte-exact).

    Args:
        docs: list of ``n_docs`` strings.
    Returns:
        sorted list of kept indices (one per distinct SHA-256 hash).
    """
    raise NotImplementedError('TODO: implement exact_dedup (see the reference in src/mlbook)')

def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity ``|a ∩ b| / |a ∪ b|`` (0.0 when both empty)."""
    raise NotImplementedError('TODO: implement jaccard (see the reference in src/mlbook)')

@dataclass
class MinHash:
    """MinHash signatures with ``num_perm`` universal hash functions.

    Each hash is ``h_i(x) = ((a_i * x + b_i) mod p) mod 2^32`` with ``p`` a
    Mersenne prime.  The signature of a set is the element-wise minimum over
    the set's shingle hashes, shape ``(num_perm,)``.
    """
    num_perm: int = 128
    seed: int = 0
    a: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        raise NotImplementedError('TODO: implement __post_init__ (see the reference in src/mlbook)')

    def signature(self, shingle_set: set[str]) -> np.ndarray:
        """MinHash signature of one document.

        Args:
            shingle_set: set of ``n_shingles`` strings.
        Returns:
            (num_perm,) uint64 array of per-permutation minimum hashes.
        """
        raise NotImplementedError('TODO: implement signature (see the reference in src/mlbook)')

def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of agreeing signature slots, an unbiased estimate of Jaccard.

    Args:
        sig_a, sig_b: (num_perm,) signatures.
    """
    raise NotImplementedError('TODO: implement estimate_jaccard (see the reference in src/mlbook)')

class LSHIndex:
    """Banded locality-sensitive hashing over MinHash signatures.

    A signature of ``num_perm`` hashes is cut into ``bands`` bands of ``rows``
    hashes each (``num_perm = bands * rows``).  Two documents are candidates if
    *any* band matches exactly.  With Jaccard ``s`` the candidate probability is
    ``1 - (1 - s^rows)^bands``, an S-curve with threshold ~``(1/bands)^(1/rows)``.
    """

    def __init__(self, num_perm: int, bands: int) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        raise NotImplementedError('TODO: implement _band_keys (see the reference in src/mlbook)')

    def add(self, doc_id: int, sig: np.ndarray) -> None:
        """Insert a (num_perm,) signature under ``doc_id``."""
        raise NotImplementedError('TODO: implement add (see the reference in src/mlbook)')

    def query(self, sig: np.ndarray) -> set[int]:
        """All doc ids sharing at least one band with ``sig`` (candidate pairs)."""
        raise NotImplementedError('TODO: implement query (see the reference in src/mlbook)')

    def threshold(self) -> float:
        """Jaccard value at which the S-curve crosses 0.5, ``(1/bands)^(1/rows)``."""
        raise NotImplementedError('TODO: implement threshold (see the reference in src/mlbook)')

def near_dedup(docs: list[str], n: int=5, num_perm: int=128, bands: int=16, threshold: float=0.8) -> list[int]:
    """Keep one representative per near-duplicate cluster.

    Pipeline: shingle → MinHash → LSH candidates → verify estimated Jaccard ≥ threshold.
    Args:
        docs: ``n_docs`` strings.
    Returns:
        sorted kept indices (first document of each cluster wins).
    """
    raise NotImplementedError('TODO: implement near_dedup (see the reference in src/mlbook)')

def ngram_contamination(train_docs: list[str], eval_docs: list[str], n: int=13) -> dict:
    """GPT-3 / Llama style n-gram overlap contamination check.

    An evaluation document is *contaminated* if any of its word n-grams occurs
    anywhere in the training corpus.  Returns the fraction of contaminated eval
    documents and their indices.  ``n=13`` was used by GPT-3; Llama 3 uses
    token-level 8-grams with a per-benchmark threshold on the overlap ratio.
    """
    raise NotImplementedError('TODO: implement ngram_contamination (see the reference in src/mlbook)')
