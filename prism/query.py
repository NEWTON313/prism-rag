"""Transformations de requête (leçon 3.3) : la question de l'utilisateur n'est pas la meilleure requête.

    rewrite     : reformuler dans le vocabulaire des documents
    hyde        : rédiger une réponse HYPOTHÉTIQUE et chercher avec elle (Hypothetical Document Embeddings)
    multi_query : plusieurs formulations, résultats fusionnés par RRF
    decompose   : une question composée → des sous-questions, une recherche chacune
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from . import llm
from .retrieval import rrf
from .vector import Hit

REWRITE = """Reformule la question d'un client en une requête de recherche pour le centre d'aide
d'une boutique en ligne : vocabulaire précis des documents (rétractation, garantie, frais de remise
en stock…), sans politesse, en français même si la question est dans une autre langue.
Réponds uniquement par la requête.

Question : {question}"""

HYDE = """Rédige en trois phrases le paragraphe du centre d'aide d'une boutique en ligne qui
répondrait à cette question. Invente des valeurs plausibles si nécessaire : ce texte sert
uniquement à la recherche, jamais à répondre au client.

Question : {question}"""


class Queries(BaseModel):
    queries: list[str] = Field(description="Formulations différentes de la même question, en français")


class SubQuestions(BaseModel):
    questions: list[str] = Field(description="Sous-questions autonomes ; une seule si la question est simple")


def rewrite(question: str) -> str:
    return llm.complete("rewrite", REWRITE.format(question=question), fast=True, max_tokens=200)


def hyde(question: str) -> str:
    return llm.complete("hyde", HYDE.format(question=question), fast=True, max_tokens=300)


def multi_query(question: str, n: int = 3) -> list[str]:
    prompt = f"Propose {n} formulations différentes de cette question pour un moteur de recherche.\n\nQuestion : {question}"
    return llm.structured("multi_query", Queries, prompt).queries[:n]


def decompose(question: str) -> list[str]:
    prompt = ("Si cette question porte sur plusieurs sujets, découpe-la en sous-questions autonomes "
              f"(trois au plus). Sinon, renvoie-la telle quelle.\n\nQuestion : {question}")
    return llm.structured("decompose", SubQuestions, prompt).questions[:3] or [question]


def search_multi(retriever, question: str, k: int = 5, n: int = 3) -> list[Hit]:
    """Recherche avec la question d'origine ET ses reformulations, fusion par RRF."""
    queries = [question, *multi_query(question, n)]
    return rrf([retriever.search(q, 20) for q in queries], limit=k)


def search_hyde(dense_retriever, question: str, k: int = 5) -> list[Hit]:
    """HyDE : on cherche avec l'embedding d'une réponse imaginaire, plus proche des documents
    que la question."""
    return dense_retriever.search(hyde(question), k)


def search_decomposed(retriever, question: str, k: int = 5) -> list[Hit]:
    """Une recherche par sous-question ; la fusion garantit que chaque sujet est représenté."""
    return rrf([retriever.search(q, 10) for q in decompose(question)], limit=k)
