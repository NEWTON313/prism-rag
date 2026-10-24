"""Re-ranking (leçon 3.2) : relire question ET passage ensemble, puis re-classer.

Un bi-encodeur (embedding) encode question et passage SÉPARÉMENT : rapide, mais il compare deux
résumés. Un cross-encodeur lit la paire complète : bien plus précis, mais trop lent pour tout le corpus.
D'où le motif : retrouver large (20 à 100 candidats), re-classer finement, garder les meilleurs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from . import config
from .embeddings import WORD
from .vector import Hit


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]: ...


class CrossEncoderReranker:
    def __init__(self, model: str = config.RERANK_MODEL) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self._model = TextCrossEncoder(model, cache_dir=str(config.CACHE_DIR / "models"))

    def rerank(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
        scores = list(self._model.rerank(query, [h.chunk.indexed_text() for h in hits]))
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:k]
        return [Hit(h.chunk, float(s)) for h, s in ranked]


class OverlapReranker:
    """Factice : nombre de racines de mots communes entre question et passage."""

    name = "overlap"

    def rerank(self, query: str, hits: list[Hit], k: int = 5) -> list[Hit]:
        q = {w[:6] for w in WORD.findall(query.lower()) if len(w) > 2}
        scored = [(h, len(q & {w[:6] for w in WORD.findall(h.chunk.indexed_text().lower())}) + h.score * 1e-3)
                  for h in hits]
        return [Hit(h.chunk, float(s)) for h, s in sorted(scored, key=lambda x: x[1], reverse=True)[:k]]


@lru_cache(maxsize=2)
def default_reranker() -> Reranker:
    return OverlapReranker() if config.FAKE else CrossEncoderReranker()
