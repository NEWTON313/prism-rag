"""Corpus, BM25, index vectoriel et quantization, fusion, re-ranking, métriques. Modules 1 à 3."""

import numpy as np
import pytest

from prism import corpus, metrics
from prism.bm25 import BM25, tokenize
from prism.corpus import Chunk
from prism.rerank import OverlapReranker
from prism.retrieval import rrf, weighted_fusion
from prism.vector import Hit, VectorIndex


def test_corpus_and_questions(articles):
    questions = corpus.load_questions()
    ids = {a.id for a in articles}
    assert len(articles) == 36 and len(questions) == 45
    assert all(set(q.relevant) <= ids for q in questions)  # chaque article attendu existe
    assert sum(not q.relevant for q in questions) == 5  # questions hors base


def test_chunks_keep_their_article(chunks):
    assert all(c.id.startswith(c.article_id + "#") for c in chunks)
    assert all(c.parent_id == c.article_id for c in chunks)


def test_tokenizer_normalizes():
    assert tokenize("Garanties ÉTENDUES du bureau") == ["garant", "etendu", "bureau"]
    assert "e02" in tokenize("Code E02 sur le DESK-2")
    assert "desk-2" in tokenize("Code E02 sur le DESK-2")  # références conservées entières


def test_bm25_rewards_rare_terms():
    bm25 = BM25(["le bureau affiche le code E02", "le bureau est livré monté", "la chaise est livrée"])
    scores = bm25.scores("code E02 bureau")
    assert scores[0] > scores[1] > scores[2] == 0


def _index(quantization):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(200, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = [Chunk(id=f"c{i}", article_id=f"a{i % 20}", category="x" if i % 2 else "y", title="", text="") for i in range(200)]
    return VectorIndex(chunks, vectors, quantization), vectors


def test_quantization_memory_and_quality():
    exact, vectors = _index("float32")
    int8, _ = _index("int8")
    binary, _ = _index("binary")
    assert exact.memory_bytes() == 4 * int8.memory_bytes() == 32 * binary.memory_bytes()
    query = vectors[0] + 0.1 * vectors[1]
    query /= np.linalg.norm(query)
    truth = {h.chunk.id for h in exact.search(query, 10)}
    assert len(truth & {h.chunk.id for h in int8.search(query, 10)}) >= 9
    rescored = {h.chunk.id for h in binary.search(query, 10, rescore=50)}
    assert len(truth & rescored) >= len(truth & {h.chunk.id for h in binary.search(query, 10)})


def test_filter_is_applied_before_ranking():
    index, vectors = _index("float32")
    hits = index.search(vectors[0], 5, where=lambda c: c.category == "x")
    assert len(hits) == 5 and all(h.chunk.category == "x" for h in hits)


def _hit(cid, score):
    return Hit(Chunk(id=cid, article_id=cid, category="", title="", text=""), score)


def test_rrf_uses_ranks_not_scores():
    dense = [_hit("a", 0.91), _hit("b", 0.90), _hit("c", 0.20)]
    lexical = [_hit("c", 25.0), _hit("a", 3.0)]
    fused = [h.chunk.id for h in rrf([dense, lexical])]
    assert fused[0] == "a"  # bien classé dans les deux listes
    assert fused.index("c") < fused.index("b")  # présent dans les deux listes, même mal classé dans l'une


def test_weighted_fusion_normalizes_scores():
    fused = weighted_fusion([_hit("a", 0.9), _hit("b", 0.1)], [_hit("b", 30.0), _hit("a", 10.0)], alpha=0.5)
    assert {h.chunk.id for h in fused} == {"a", "b"}


def test_hybrid_finds_both_lexical_and_semantic(hybrid, lexical, dense):
    code = [h.chunk.article_id for h in hybrid.search("Que signifie le code E02 ?", 5)]
    assert "desk-02" in code


def test_reranker_reorders():
    hits = [_hit("x", 0.9), _hit("y", 0.8)]
    hits[0] = Hit(Chunk(id="x", article_id="x", category="", title="Livraison", text="Tarifs express"), 0.9)
    hits[1] = Hit(Chunk(id="y", article_id="y", category="", title="Retours", text="Délai de rétractation de 30 jours"), 0.8)
    assert OverlapReranker().rerank("délai de rétractation", hits, 2)[0].chunk.id == "y"


def test_metrics():
    ranked = ["a", "b", "c", "d"]
    assert metrics.recall_at_k(ranked, ["b", "z"], 3) == 0.5
    assert metrics.reciprocal_rank(ranked, ["c"]) == pytest.approx(1 / 3)
    assert metrics.ndcg_at_k(ranked, ["a"], 5) == 1.0
    assert metrics.ndcg_at_k(ranked, ["b"], 5) < 1.0


def test_evaluate_retrieval_report(hybrid):
    report = metrics.evaluate_retrieval("hybride", lambda q: hybrid.search(q, 10), corpus.load_questions())
    assert 0 < report.recall <= 1 and set(report.by_type) == {"anglais", "factuelle", "lexicale", "multi", "reformulation"}
