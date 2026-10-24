"""Jeux de tests synthétiques fiables (leçon 5.2).

Générer des questions à partir des passages est facile ; en générer de BONNES l'est moins. Trois filtres :
1. répondable : la question a une réponse dans son passage source (vérifiée par un second appel) ;
2. pas recopiée : trop de mots communs avec le passage → question trop facile pour la recherche lexicale ;
3. pas de doublon : deux questions quasi identiques faussent les moyennes.
Puis une relecture humaine d'un échantillon, toujours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from . import llm
from .bm25 import tokenize
from .corpus import Chunk
from .embeddings import Embedder


class GeneratedQuestion(BaseModel):
    question: str = Field(description="Question qu'un VRAI client poserait, avec ses propres mots, "
                                      "sans recopier le texte")
    answer: str = Field(description="Réponse courte, tirée du passage")
    style: Literal["simple", "reformulee", "familiere"] = Field(description="Registre de la question")


class Answerable(BaseModel):
    answerable: bool = Field(description="True si le passage suffit à répondre complètement à la question")


@dataclass
class TestCase:
    question: str
    answer: str
    source_chunk: str
    source_article: str
    style: str


def lexical_overlap(question: str, passage: str) -> float:
    q = set(tokenize(question))
    return len(q & set(tokenize(passage))) / len(q) if q else 1.0


def generate_testset(chunks: list[Chunk], embedder: Embedder, max_overlap: float = 0.6,
                     dedup_threshold: float = 0.92) -> tuple[list[TestCase], dict[str, int]]:
    rejected = {"non_repondable": 0, "trop_recopiee": 0, "doublon": 0}
    kept: list[TestCase] = []
    for chunk in chunks:
        prompt = f"Passage du centre d'aide ({chunk.title}) :\n{chunk.text}\n\nRédige une question de client."
        g = llm.structured("testset_generate", GeneratedQuestion, prompt)
        check_prompt = f"Passage : {chunk.text}\n\nQuestion : {g.question}"
        check = llm.structured("testset_check", Answerable, check_prompt)
        if not check.answerable:
            rejected["non_repondable"] += 1
            continue
        if lexical_overlap(g.question, chunk.text) > max_overlap:
            rejected["trop_recopiee"] += 1
            continue
        kept.append(TestCase(g.question, g.answer, chunk.id, chunk.article_id, g.style))
    if kept:  # déduplication par similarité des embeddings
        vectors = embedder.embed([t.question for t in kept])
        unique: list[int] = []
        for i in range(len(kept)):
            if all(float(vectors[i] @ vectors[j]) < dedup_threshold for j in unique):
                unique.append(i)
        rejected["doublon"] = len(kept) - len(unique)
        kept = [kept[i] for i in unique]
    return kept, rejected


def overlap_profile(cases: list[TestCase], passages: dict[str, str]) -> float:
    """Recouvrement lexical moyen question/passage : comparez-le à celui de vos VRAIES questions."""
    return float(np.mean([lexical_overlap(c.question, passages[c.source_chunk]) for c in cases])) if cases else 0.0
