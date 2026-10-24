"""Métriques de recherche (leçons 1.2 et 5.1), calculées au niveau de l'ARTICLE.

recall@k : part des articles pertinents retrouvés dans les k premiers résultats
MRR      : moyenne de 1/rang du premier article pertinent
nDCG@k   : gain cumulé actualisé normalisé — récompense les bons résultats placés haut
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .corpus import Question
from .vector import Hit


def article_ranking(hits: list[Hit]) -> list[str]:
    """Liste des articles dans l'ordre de leur premier passage retrouvé (sans doublon)."""
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.article_id not in seen:
            seen.append(hit.chunk.article_id)
    return seen


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float:
    return next((1 / i for i, a in enumerate(ranked, start=1) if a in relevant), 0.0)


def ndcg_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, a in enumerate(ranked[:k], start=1) if a in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


@dataclass
class RetrievalReport:
    name: str
    recall: float
    mrr: float
    ndcg: float
    by_type: dict[str, float]  # recall@k par type de question
    failures: list[str]


def evaluate_retrieval(name: str, search, questions: list[Question], k: int = 5) -> RetrievalReport:
    """`search(question) -> list[Hit]`. Les questions hors base sont exclues (pas d'article attendu)."""
    answerable = [q for q in questions if q.relevant]
    recalls, rrs, ndcgs, by_type, failures = [], [], [], defaultdict(list), []
    for q in answerable:
        ranked = article_ranking(search(q.question))
        r = recall_at_k(ranked, q.relevant, k)
        recalls.append(r)
        rrs.append(reciprocal_rank(ranked, q.relevant))
        ndcgs.append(ndcg_at_k(ranked, q.relevant, k))
        by_type[q.type].append(r)
        if r == 0:
            failures.append(f"{q.id} [{q.type}] {q.question}")
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    return RetrievalReport(name, mean(recalls), mean(rrs), mean(ndcgs),
                           {t: mean(v) for t, v in sorted(by_type.items())}, failures)
