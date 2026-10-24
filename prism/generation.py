"""Génération de la réponse à partir des passages retrouvés, avec citations."""

from __future__ import annotations

import re

from . import llm
from .vector import Hit

ANSWER = """Réponds à la question du client UNIQUEMENT à partir des extraits du centre d'aide ci-dessous.
Cite l'identifiant de l'article après chaque information, entre crochets, par exemple [ret-01].
Si les extraits ne contiennent pas la réponse, dis-le : le client prendra des décisions sur cette base.
{strict}
Extraits :
{context}

Question : {question}"""

STRICT = "N'écris AUCUNE information absente des extraits, même si elle te semble évidente.\n"
CITATION = re.compile(r"\[([a-z]+-\d+)\]")


def format_context(hits: list[Hit]) -> str:
    return "\n".join(f"[{h.chunk.article_id}] {h.chunk.title} : {h.chunk.text}" for h in hits)


def answer_with_citations(question: str, hits: list[Hit], strict: bool = False) -> tuple[str, list[str]]:
    """Renvoie la réponse et les articles RÉELLEMENT cités (pas seulement fournis au modèle)."""
    answer = llm.complete("answer", ANSWER.format(context=format_context(hits), question=question,
                                                  strict=STRICT if strict else ""))
    provided = {h.chunk.article_id for h in hits}
    cited = [c for c in dict.fromkeys(CITATION.findall(answer)) if c in provided]
    return answer, cited
