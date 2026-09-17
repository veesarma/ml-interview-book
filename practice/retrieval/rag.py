# mlbook-practice-stub: untouched
# Practice stub for src/mlbook/retrieval/rag.py
# Fill in every `raise NotImplementedError`, then run:
#     MLBOOK_IMPL=practice pytest tests/ -k rag -q
# Regenerate a clean stub with:
#     python scripts/make_practice_stubs.py retrieval/rag --force

"""A small, fully offline retrieval-augmented generation (RAG) pipeline.

    query -> [dense (brute force or IVF) + BM25] -> RRF fusion -> metadata filter
          -> reranker -> prompt with numbered citations -> deterministic generator

Everything here is a toy by design: the embedder hashes character n-grams, the
reranker scores lexical overlap, and the generator is extractive. The *structure*
is the production structure; the models are stand-ins so the tests run in
milliseconds with no network.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from mlbook.retrieval.bm25 import BM25, tokenize
from mlbook.retrieval.hybrid import reciprocal_rank_fusion
from mlbook.retrieval.ivf import IVFIndex
from mlbook.retrieval.similarity import brute_force_topk
from mlbook.retrieval.toy_embedder import HashNGramEmbedder

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    metadata: dict = field(default_factory=dict)

def chunk_fixed(text: str, size: int=200, overlap: int=50) -> list[str]:
    """Fixed-size character windows with overlap. stride = size - overlap."""
    raise NotImplementedError('TODO: implement chunk_fixed (see the reference in src/mlbook)')

def chunk_structural(text: str) -> list[str]:
    """Split on blank lines (paragraphs / OCR blocks); keeps natural boundaries."""
    raise NotImplementedError('TODO: implement chunk_structural (see the reference in src/mlbook)')

def chunk_sentences(text: str, max_chars: int=300) -> list[str]:
    """Greedy sentence packing: add sentences until max_chars would be exceeded."""
    raise NotImplementedError('TODO: implement chunk_sentences (see the reference in src/mlbook)')

def overlap_reranker(query: str, chunks: list[Chunk]) -> np.ndarray:
    """Stand-in cross-encoder: fraction of query tokens present in each chunk. -> (n,)."""
    raise NotImplementedError('TODO: implement overlap_reranker (see the reference in src/mlbook)')

def extractive_generator(query: str, context: list[Chunk]) -> tuple[str, list[int]]:
    """Deterministic 'LLM': returns the sentence with the most query-token overlap
    and cites the 1-based index of the chunk it came from."""
    raise NotImplementedError('TODO: implement extractive_generator (see the reference in src/mlbook)')

def compose_prompt(query: str, context: list[Chunk]) -> str:
    """Numbered context blocks + instruction to cite. Citations are [i]."""
    raise NotImplementedError('TODO: implement compose_prompt (see the reference in src/mlbook)')

class RAGPipeline:
    """Hybrid retrieval + rerank + cite. See module docstring for the data flow."""

    def __init__(self, embedder: HashNGramEmbedder | None=None, chunker: Callable[[str], list[str]]=chunk_structural, reranker: Callable[[str, list[Chunk]], np.ndarray]=overlap_reranker, generator: Callable[[str, list[Chunk]], tuple[str, list[int]]]=extractive_generator, use_ivf: bool=False, n_list: int=4, rrf_k: int=60) -> None:
        raise NotImplementedError('TODO: implement __init__ (see the reference in src/mlbook)')

    def index(self, docs: dict[str, str], metadata: dict[str, dict] | None=None) -> 'RAGPipeline':
        """Chunk, embed and index a {doc_id: text} corpus."""
        raise NotImplementedError('TODO: implement index (see the reference in src/mlbook)')

    def _dense(self, query: str, k: int, nprobe: int) -> list[int]:
        raise NotImplementedError('TODO: implement _dense (see the reference in src/mlbook)')

    def retrieve(self, query: str, k: int=5, k_candidates: int=20, metadata_filter: Callable[[dict], bool] | None=None, nprobe: int=2) -> list[Chunk]:
        """Hybrid candidates -> RRF -> post-filter -> rerank -> top-k chunks."""
        raise NotImplementedError('TODO: implement retrieve (see the reference in src/mlbook)')

    def answer(self, query: str, k: int=3, **retrieve_kwargs) -> dict:
        """Full pipeline. Returns answer text, cited chunks, prompt and retrieved chunks."""
        raise NotImplementedError('TODO: implement answer (see the reference in src/mlbook)')

def retrieval_recall_at_k(retrieved: list[Chunk], relevant_doc_ids: set[str]) -> float:
    """Fraction of relevant documents that have at least one chunk in the top-k."""
    raise NotImplementedError('TODO: implement retrieval_recall_at_k (see the reference in src/mlbook)')

def citation_precision(answer: dict, relevant_doc_ids: set[str]) -> float:
    """Fraction of cited chunks that come from a relevant document."""
    raise NotImplementedError('TODO: implement citation_precision (see the reference in src/mlbook)')

def faithfulness_overlap(answer_text: str, context: list[Chunk]) -> float:
    """Lexical stand-in for faithfulness: share of answer tokens that appear in the
    context (1.0 = every claim token is grounded). Real systems use an NLI judge."""
    raise NotImplementedError('TODO: implement faithfulness_overlap (see the reference in src/mlbook)')
