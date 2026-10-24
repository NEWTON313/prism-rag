"""RAG auto-correctif (leçon 4.3) : CRAG et vérification à la Self-RAG.

CRAG (Corrective RAG) : un évaluateur juge chaque passage retrouvé AVANT la génération.
    pertinents        → on répond avec eux, après en avoir retiré les phrases inutiles
    aucun pertinent   → on reformule et on recherche à nouveau (une fois), sinon on avoue ne pas savoir
Self-RAG (simplifié) : après la génération, on vérifie que chaque affirmation est soutenue par les passages ;
sinon, on régénère avec une consigne plus stricte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from . import llm, query
from .generation import answer_with_citations
from .vector import Hit


class Grade(BaseModel):
    verdict: Literal["pertinent", "partiel", "hors_sujet"] = Field(
        description="pertinent : répond à la question ; partiel : utile mais incomplet ; "
                    "hors_sujet : inutile")


class SupportCheck(BaseModel):
    supported: bool = Field(description="True si TOUTES les affirmations de la réponse figurent dans les extraits")
    unsupported: list[str] = Field(description="Affirmations absentes des extraits, le cas échéant")


@dataclass
class CorrectiveResult:
    answer: str
    sources: list[str]
    actions: list[str] = field(default_factory=list)


def grade(question: str, hit: Hit) -> str:
    prompt = f"Question : {question}\n\nPassage : {hit.chunk.title} — {hit.chunk.text}\n\nCe passage aide-t-il à répondre ?"
    return llm.structured("grade", Grade, prompt).verdict


def corrective_answer(retriever, question: str, k: int = 5) -> CorrectiveResult:
    actions: list[str] = []
    hits = retriever.search(question, k)
    kept = [h for h in hits if grade(question, h) != "hors_sujet"]
    actions.append(f"évaluation : {len(kept)}/{len(hits)} passages gardés")
    if not kept:
        rewritten = query.rewrite(question)
        actions.append(f"reformulation : {rewritten}")
        hits = retriever.search(rewritten, k)
        kept = [h for h in hits if grade(question, h) != "hors_sujet"]
        actions.append(f"nouvelle évaluation : {len(kept)}/{len(hits)} passages gardés")
    if not kept:
        return CorrectiveResult("Je ne trouve pas cette information dans la base.", [], actions)

    answer, sources = answer_with_citations(question, kept)
    context = "\n".join(f"[{h.chunk.article_id}] {h.chunk.text}" for h in kept)
    check = llm.structured("support_check", SupportCheck,
                           f"Extraits :\n{context}\n\nRéponse à vérifier :\n{answer}")
    if not check.supported:
        actions.append(f"affirmations non soutenues : {check.unsupported} → régénération stricte")
        answer, sources = answer_with_citations(question, kept, strict=True)
    return CorrectiveResult(answer, sources, actions)
