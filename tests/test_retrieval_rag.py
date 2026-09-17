import numpy as np

from mlbook.retrieval import rag
from mlbook.retrieval.toy_embedder import HashNGramEmbedder

DOCS = {
    "faiss": "FAISS is a library for efficient similarity search.\n\nIt implements IVF, HNSW and product quantization on CPU and GPU.",
    "bm25": "BM25 is a sparse ranking function.\n\nIt uses term frequency saturation and document length normalization.",
    "manim": "Manim is an animation engine for explanatory math videos.\n\nScenes are written in Python.",
}
META = {"faiss": {"year": 2017}, "bm25": {"year": 1994}, "manim": {"year": 2018}}


def test_toy_embedder_deterministic_and_similar_texts_close():
    emb = HashNGramEmbedder(dim=128)
    a = emb.embed(["product quantization", "product quantisation", "animation engine"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.allclose(a[0], emb.embed_one("product quantization"))
    assert a[0] @ a[1] > a[0] @ a[2]


def test_chunk_fixed_overlap_and_coverage():
    text = "abcdefghij" * 5  # 50 chars
    chunks = rag.chunk_fixed(text, size=20, overlap=5)
    assert all(len(c) <= 20 for c in chunks)
    assert chunks[0][15:] == chunks[1][:5]  # overlap region matches
    assert "".join(c[:15] for c in chunks[:-1]) + chunks[-1] == text


def test_chunk_structural_and_sentences():
    assert rag.chunk_structural("a\n\nb\n \nc") == ["a", "b", "c"]
    s = rag.chunk_sentences("One. Two. Three four five. Six.", max_chars=10)
    assert s == ["One. Two.", "Three four five.", "Six."]


def test_overlap_reranker_and_extractive_generator():
    chunks = [rag.Chunk("d", 0, "cats purr. dogs bark."), rag.Chunk("d", 1, "fish swim.")]
    scores = rag.overlap_reranker("do dogs bark", chunks)
    assert scores[0] > scores[1]
    text, cites = rag.extractive_generator("do dogs bark", chunks)
    assert text == "dogs bark. [1]" and cites == [1]


def test_compose_prompt_numbers_context():
    p = rag.compose_prompt("q?", [rag.Chunk("x", 0, "hello")])
    assert "[1] (doc=x, chunk=0) hello" in p and p.endswith("Answer:")


def test_rag_pipeline_end_to_end_brute_force_and_ivf():
    for use_ivf in (False, True):
        pipe = rag.RAGPipeline(use_ivf=use_ivf, n_list=2).index(DOCS, META)
        out = pipe.answer("What does BM25 use for length normalization?", k=2)
        assert out["citations"] and out["citations"][0].doc_id == "bm25"
        assert rag.retrieval_recall_at_k(out["context"], {"bm25"}) == 1.0
        assert rag.citation_precision(out, {"bm25"}) == 1.0
        assert rag.faithfulness_overlap(out["answer"], out["context"]) == 1.0


def test_rag_metadata_filter_removes_docs():
    pipe = rag.RAGPipeline().index(DOCS, META)
    ctx = pipe.retrieve("similarity search library", k=3, metadata_filter=lambda m: m["year"] < 2000)
    assert ctx and all(c.doc_id == "bm25" for c in ctx)
