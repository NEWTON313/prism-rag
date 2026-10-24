"""Parent-document et contextual retrieval (leçon 3.4).

Contextual retrieval : avant d'indexer un passage, un LLM rédige une ou deux phrases qui le SITUENT dans
son document (« Extrait de la garantie de la chaise Axis, qui porte sur… »). Ce contexte est ajouté au
texte vectorisé ET au texte indexé par BM25.

Coût : un appel par passage. Le document complet est envoyé à chaque appel : on le met en CACHE (prompt
caching) — les passages suivants du même document relisent le document depuis le cache, à ~10 % du prix.
Attention : un préfixe plus court que le minimum du modèle (512 tokens pour Claude Haiku 5.5 et Opus 5.5)
n'est PAS mis en cache, sans erreur. Les articles courts de ce corpus sont sous ce seuil ; de vrais
documents de plusieurs pages en profitent pleinement.
"""

from __future__ import annotations

from dataclasses import replace

from . import llm
from .corpus import Article, Chunk
from .vector import Hit

CONTEXT_PROMPT = """Voici un passage extrait du document ci-dessus :
<passage>
{chunk}
</passage>
Rédige une à deux phrases qui situent ce passage dans le document, pour améliorer sa recherche :
de quel produit ou de quelle règle il s'agit, et dans quel cas il s'applique.
Réponds uniquement par ces phrases."""


def contextualize(chunks: list[Chunk], articles: list[Article]) -> list[Chunk]:
    by_id = {a.id: a for a in articles}
    out = []
    for chunk in chunks:
        article = by_id[chunk.article_id]
        document = f"<document>\n# {article.title}\n{article.text}\n</document>"
        context = llm.complete("contextualize", CONTEXT_PROMPT.format(chunk=chunk.text), fast=True,
                               max_tokens=150, cached_prefix=document)
        out.append(replace(chunk, meta={**chunk.meta, "context": context}))
    return out


def expand_to_parents(hits: list[Hit], articles: list[Article], k: int = 5) -> list[Hit]:
    """Parent-document retrieval : on cherche sur de petits passages, on renvoie leurs articles."""
    by_id = {a.id: a for a in articles}
    parents: list[Hit] = []
    for hit in hits:
        if any(p.chunk.article_id == hit.chunk.article_id for p in parents):
            continue  # plusieurs passages d'un même article : un seul parent
        a = by_id[hit.chunk.article_id]
        parents.append(Hit(Chunk(id=a.id, article_id=a.id, category=a.category, title=a.title, text=a.text), hit.score))
        if len(parents) == k:
            break
    return parents
