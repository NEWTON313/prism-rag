"""Retrievers : dense, lexical, hybride (fusion RRF ou pondérée) — module 3.

Tous renvoient une liste de `Hit` triés ; ils sont interchangeables dans les bancs d'essai.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .bm25 import BM25
from .corpus import Chunk
from .embeddings import Embedder
from .vector import Hit, VectorIndex


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, quantization: str = "float32") -> None:
        self.embedder = embedder
        self.index = VectorIndex(chunks, embedder.embed([c.indexed_text() for c in chunks]), quantization)

    def search(self, query: str, k: int = 5, where: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        return self.index.search(self.embedder.embed([query])[0], k, where)


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25([c.indexed_text() for c in chunks])

    def search(self, query: str, k: int = 5, where: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        scores = np.array(self.bm25.scores(query))
        if where is not None:
            scores = np.where([where(c) for c in self.chunks], scores, -np.inf)
        order = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]  # score nul : aucun mot commun
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def rrf(rankings: list[list[Hit]], k: int = 60, limit: int = 5) -> list[Hit]:
    """Reciprocal Rank Fusion : score = Σ 1 / (k + rang). Ne regarde que les RANGS, jamais les scores.

    Les scores BM25 (0 à 20…) et cosinus (−1 à 1) ne sont pas comparables : fusionner les rangs évite
    toute normalisation. k=60 est la valeur de l'article d'origine ; elle atténue l'écart entre 1er et 2e.
    """
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk.id, hit)
    order = sorted(fused, key=fused.get, reverse=True)[:limit]
    return [Hit(best[i].chunk, fused[i]) for i in order]


def weighted_fusion(dense: list[Hit], lexical: list[Hit], alpha: float = 0.5, limit: int = 5) -> list[Hit]:
    """Fusion pondérée des scores, après normalisation min-max de chaque liste : α·dense + (1−α)·lexical."""
    def normalized(hits: list[Hit]) -> dict[str, float]:
        if not hits:
            return {}
        values = [h.score for h in hits]
        lo, hi = min(values), max(values)
        return {h.chunk.id: (h.score - lo) / (hi - lo) if hi > lo else 1.0 for h in hits}

    d, l = normalized(dense), normalized(lexical)
    chunks = {h.chunk.id: h.chunk for h in dense + lexical}
    scores = {cid: alpha * d.get(cid, 0.0) + (1 - alpha) * l.get(cid, 0.0) for cid in chunks}
    order = sorted(scores, key=scores.get, reverse=True)[:limit]
    return [Hit(chunks[i], scores[i]) for i in order]


class HybridRetriever:
    def __init__(self, dense: DenseRetriever, lexical: BM25Retriever, candidates: int = 20) -> None:
        self.dense, self.lexical, self.candidates = dense, lexical, candidates

    def search(self, query: str, k: int = 5, where: Callable[[Chunk], bool] | None = None) -> list[Hit]:
        return rrf([self.dense.search(query, self.candidates, where),
                    self.lexical.search(query, self.candidates, where)], limit=k)
