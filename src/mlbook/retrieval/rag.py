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


# ----------------------------------------------------------------------------- chunking
def chunk_fixed(text: str, size: int = 200, overlap: int = 50) -> list[str]:
    """Fixed-size character windows with overlap. stride = size - overlap."""
    assert 0 <= overlap < size
    stride = size - overlap
    out = []
    for start in range(0, max(len(text), 1), stride):
        piece = text[start : start + size]
        if piece.strip():
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_structural(text: str) -> list[str]:
    """Split on blank lines (paragraphs / OCR blocks); keeps natural boundaries."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Greedy sentence packing: add sentences until max_chars would be exceeded."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    out: list[str] = []
    cur = ""
    for s in sentences:
        if cur and len(cur) + 1 + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


# ----------------------------------------------------------------------------- reranker
def overlap_reranker(query: str, chunks: list[Chunk]) -> np.ndarray:
    """Stand-in cross-encoder: fraction of query tokens present in each chunk. -> (n,)."""
    q = set(tokenize(query))
    scores = np.zeros(len(chunks))  # (n,)
    for i, c in enumerate(chunks):
        toks = set(tokenize(c.text))
        scores[i] = len(q & toks) / max(len(q), 1)
    return scores


# ----------------------------------------------------------------------------- generator
def extractive_generator(query: str, context: list[Chunk]) -> tuple[str, list[int]]:
    """Deterministic 'LLM': returns the sentence with the most query-token overlap
    and cites the 1-based index of the chunk it came from."""
    q = set(tokenize(query))
    best, best_score, best_cite = "", -1.0, []
    for i, c in enumerate(context, start=1):
        for s in re.split(r"(?<=[.!?])\s+", c.text):
            score = len(q & set(tokenize(s)))
            if score > best_score:
                best, best_score, best_cite = s.strip(), score, [i]
    return f"{best} [{best_cite[0]}]" if best else "I could not find this in the provided context.", best_cite


def compose_prompt(query: str, context: list[Chunk]) -> str:
    """Numbered context blocks + instruction to cite. Citations are [i]."""
    lines = ["Answer the question using only the context. Cite sources as [i].", ""]
    for i, c in enumerate(context, start=1):
        lines.append(f"[{i}] (doc={c.doc_id}, chunk={c.chunk_id}) {c.text}")
    lines += ["", f"Question: {query}", "Answer:"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- pipeline
class RAGPipeline:
    """Hybrid retrieval + rerank + cite. See module docstring for the data flow."""

    def __init__(
        self,
        embedder: HashNGramEmbedder | None = None,
        chunker: Callable[[str], list[str]] = chunk_structural,
        reranker: Callable[[str, list[Chunk]], np.ndarray] = overlap_reranker,
        generator: Callable[[str, list[Chunk]], tuple[str, list[int]]] = extractive_generator,
        use_ivf: bool = False,
        n_list: int = 4,
        rrf_k: int = 60,
    ) -> None:
        self.embedder = embedder or HashNGramEmbedder()
        self.chunker = chunker
        self.reranker = reranker
        self.generator = generator
        self.use_ivf = use_ivf
        self.n_list = n_list
        self.rrf_k = rrf_k
        self.chunks: list[Chunk] = []
        self.E: np.ndarray | None = None  # (N, dim)
        self.bm25 = BM25()
        self.ivf: IVFIndex | None = None

    def index(self, docs: dict[str, str], metadata: dict[str, dict] | None = None) -> "RAGPipeline":
        """Chunk, embed and index a {doc_id: text} corpus."""
        metadata = metadata or {}
        for doc_id, text in docs.items():
            for j, piece in enumerate(self.chunker(text)):
                self.chunks.append(Chunk(doc_id, j, piece, dict(metadata.get(doc_id, {}))))
        texts = [c.text for c in self.chunks]
        self.E = self.embedder.embed(texts)  # (N, dim)
        self.bm25.fit(texts)
        if self.use_ivf:
            n_list = min(self.n_list, len(self.chunks))
            self.ivf = IVFIndex(n_list=n_list).train(self.E).add(self.E)
        return self

    def _dense(self, query: str, k: int, nprobe: int) -> list[int]:
        q = self.embedder.embed([query])  # (1, dim)
        if self.ivf is not None:
            ids, _ = self.ivf.search(q[0], k, nprobe=nprobe)  # (k',)
            return ids.tolist()
        ids, _ = brute_force_topk(q, self.E, k, metric="dot")  # (1, k)
        return ids[0].tolist()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        k_candidates: int = 20,
        metadata_filter: Callable[[dict], bool] | None = None,
        nprobe: int = 2,
    ) -> list[Chunk]:
        """Hybrid candidates -> RRF -> post-filter -> rerank -> top-k chunks."""
        assert self.E is not None, "call index() first"
        dense_ids = self._dense(query, k_candidates, nprobe)
        sparse_ids, _ = self.bm25.topk(query, k_candidates)
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids.tolist()], k=self.rrf_k)
        cands = [self.chunks[i] for i, _ in fused]
        if metadata_filter is not None:
            cands = [c for c in cands if metadata_filter(c.metadata)]
        if not cands:
            return []
        scores = self.reranker(query, cands)  # (n_cand,)
        order = np.argsort(-scores, kind="stable")[:k]  # (k,)
        return [cands[i] for i in order]

    def answer(self, query: str, k: int = 3, **retrieve_kwargs) -> dict:
        """Full pipeline. Returns answer text, cited chunks, prompt and retrieved chunks."""
        context = self.retrieve(query, k=k, **retrieve_kwargs)
        prompt = compose_prompt(query, context)
        text, cites = self.generator(query, context)
        cited = [context[i - 1] for i in cites if 1 <= i <= len(context)]
        return {"answer": text, "citations": cited, "context": context, "prompt": prompt}


# ----------------------------------------------------------------------------- evaluation
def retrieval_recall_at_k(retrieved: list[Chunk], relevant_doc_ids: set[str]) -> float:
    """Fraction of relevant documents that have at least one chunk in the top-k."""
    hit = {c.doc_id for c in retrieved} & relevant_doc_ids
    return len(hit) / max(len(relevant_doc_ids), 1)


def citation_precision(answer: dict, relevant_doc_ids: set[str]) -> float:
    """Fraction of cited chunks that come from a relevant document."""
    cited = answer["citations"]
    if not cited:
        return 0.0
    return sum(c.doc_id in relevant_doc_ids for c in cited) / len(cited)


def faithfulness_overlap(answer_text: str, context: list[Chunk]) -> float:
    """Lexical stand-in for faithfulness: share of answer tokens that appear in the
    context (1.0 = every claim token is grounded). Real systems use an NLI judge."""
    ans = [t for t in tokenize(answer_text) if not t.isdigit()]
    ctx = set()
    for c in context:
        ctx.update(tokenize(c.text))
    return sum(t in ctx for t in ans) / max(len(ans), 1)
