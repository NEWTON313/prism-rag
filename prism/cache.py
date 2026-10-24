"""Cache sémantique (leçon 5.3) : ne pas recalculer une réponse à une question déjà posée… autrement.

Une question est servie depuis le cache si une question passée est assez SIMILAIRE (cosinus ≥ seuil),
dans le même PÉRIMÈTRE (langue, client, droits d'accès…) et si ses sources n'ont pas changé depuis.
Le seuil est le réglage critique : trop bas, on sert la réponse d'une autre question.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .embeddings import Embedder


@dataclass
class CacheEntry:
    question: str
    vector: np.ndarray
    answer: str
    sources: list[str]
    scope: str
    created: float = field(default_factory=time.time)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SemanticCache:
    def __init__(self, embedder: Embedder, threshold: float = 0.92, ttl_seconds: float = 24 * 3600) -> None:
        self.embedder, self.threshold, self.ttl = embedder, threshold, ttl_seconds
        self.entries: list[CacheEntry] = []
        self.stats = CacheStats()

    def lookup(self, question: str, scope: str = "default") -> CacheEntry | None:
        vector = self.embedder.embed([question])[0]
        now = time.time()
        best, best_score = None, -1.0
        for entry in self.entries:
            if entry.scope != scope or now - entry.created > self.ttl:
                continue
            score = float(entry.vector @ vector)
            if score > best_score:
                best, best_score = entry, score
        if best is not None and best_score >= self.threshold:
            self.stats.hits += 1
            return best
        self.stats.misses += 1
        return None

    def store(self, question: str, answer: str, sources: list[str], scope: str = "default") -> None:
        self.entries.append(CacheEntry(question, self.embedder.embed([question])[0], answer, sources, scope))

    def invalidate_sources(self, changed: set[str]) -> int:
        """Un article modifié invalide toutes les réponses qui le citaient."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if not set(e.sources) & changed]
        return before - len(self.entries)
