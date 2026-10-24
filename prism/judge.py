"""Métriques de génération avec un LLM juge (leçon 5.1) — les définitions de RAGAS, écrites en clair.

faithfulness      : part des affirmations de la RÉPONSE soutenues par les passages (détecte les inventions)
context_recall    : part des affirmations de la RÉFÉRENCE présentes dans les passages (la recherche a-t-elle
                    trouvé tout ce qu'il fallait ?)
context_precision : les passages utiles sont-ils classés en tête ? (moyenne des précisions aux rangs utiles)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import llm
from .generation import CITATION


class Statements(BaseModel):
    statements: list[str] = Field(description="Affirmations atomiques, autonomes, une idée par affirmation")


class Verdicts(BaseModel):
    supported: list[bool] = Field(description="Pour chaque affirmation, dans l'ordre : figure-t-elle dans les extraits ?")


class Useful(BaseModel):
    useful: bool = Field(description="True si ce passage contient une information utile pour la réponse de référence")


def _statements(text: str) -> list[str]:
    prompt = f"Découpe ce texte en affirmations atomiques et autonomes :\n{text}"
    return llm.structured("judge_statements", Statements, prompt).statements


def _supported(statements: list[str], contexts: list[str]) -> list[bool]:
    if not statements:
        return []
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    prompt = "Extraits :\n" + "\n".join(contexts) + f"\n\nAffirmations :\n{numbered}\n\nChacune figure-t-elle dans les extraits ?"
    verdicts = llm.structured("judge_support", Verdicts, prompt).supported
    return (verdicts + [False] * len(statements))[: len(statements)]


def faithfulness(answer: str, contexts: list[str]) -> float:
    # une citation [ret-01] n'est pas une affirmation
    statements = _statements(CITATION.sub("", answer))
    verdicts = _supported(statements, contexts)
    return sum(verdicts) / len(verdicts) if verdicts else 1.0


def context_recall(reference: str, contexts: list[str]) -> float:
    statements = _statements(reference)
    verdicts = _supported(statements, contexts)
    return sum(verdicts) / len(verdicts) if verdicts else 1.0


def context_precision(question: str, reference: str, contexts: list[str]) -> float:
    useful = [llm.structured("judge_useful", Useful,
                             f"Question : {question}\nRéponse de référence : {reference}\n\nPassage : {c}").useful
              for c in contexts]
    if not any(useful):
        return 0.0
    precisions = [sum(useful[: i + 1]) / (i + 1) for i, u in enumerate(useful) if u]
    return sum(precisions) / len(precisions)
