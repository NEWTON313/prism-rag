"""Banc d'essai : chaque technique du cours, mesurée sur le même jeu de questions (leçons 1.2 et 3.x).

    python -m prism.bench              # embeddings et re-ranker réels (FastEmbed, local, gratuit)
    PRISM_FAKE=1 python -m prism.bench  # hors ligne, chiffres sans valeur : vérifie seulement la mécanique
"""

from __future__ import annotations

import sys
import time

from . import corpus, metrics, retrieval
from .embeddings import default_embedder
from .rerank import default_reranker


def build(k_candidates: int = 20):
    articles = corpus.load_articles()
    chunks = corpus.chunk_articles(articles)
    embedder = default_embedder()
    dense = retrieval.DenseRetriever(chunks, embedder)
    lexical = retrieval.BM25Retriever(chunks)
    hybrid = retrieval.HybridRetriever(dense, lexical, k_candidates)
    return articles, chunks, embedder, dense, lexical, hybrid


def configurations(with_rerank: bool = True):
    articles, chunks, embedder, dense, lexical, hybrid = build()
    configs = {
        "BM25": lambda q: lexical.search(q, 10),
        "Dense": lambda q: dense.search(q, 10),
        "Hybride (RRF)": lambda q: hybrid.search(q, 10),
    }
    for quantization in ("int8", "binary"):
        retriever = retrieval.DenseRetriever(chunks, embedder, quantization)
        configs[f"Dense {quantization}"] = (lambda r: lambda q: r.search(q, 10))(retriever)
    binary = retrieval.DenseRetriever(chunks, embedder, "binary")
    configs["Dense binary + re-scoring"] = lambda q: binary.index.search(embedder.embed([q])[0], 10, rescore=30)
    if with_rerank:
        reranker = default_reranker()
        configs["Dense + re-ranking"] = lambda q: reranker.rerank(q, dense.search(q, 20), 10)
        configs["Hybride + re-ranking"] = lambda q: reranker.rerank(q, hybrid.search(q, 20), 10)
    return configs


def run(with_rerank: bool = True, k: int = 5) -> list[tuple[metrics.RetrievalReport, float]]:
    questions = corpus.load_questions()
    results = []
    for name, search in configurations(with_rerank).items():
        start = time.perf_counter()
        report = metrics.evaluate_retrieval(name, search, questions, k)
        results.append((report, (time.perf_counter() - start) * 1000 / len([q for q in questions if q.relevant])))
    return results


def print_table(results) -> None:
    types = list(results[0][0].by_type)
    print(f"{'Configuration':<28}{'R@5':>6}{'MRR':>6}{'nDCG':>6}" + "".join(f"{t[:9]:>11}" for t in types) + f"{'ms/q':>8}")
    for report, ms in results:
        print(f"{report.name:<28}{report.recall:>6.2f}{report.mrr:>6.2f}{report.ndcg:>6.2f}"
              + "".join(f"{report.by_type[t]:>11.2f}" for t in types) + f"{ms:>8.1f}")


if __name__ == "__main__":
    print_table(run(with_rerank="--no-rerank" not in sys.argv))
