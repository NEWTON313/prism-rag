"""Leçons 3.3 et 3.4 — Transformations de requête, contextual retrieval, parent-document.

    PRISM_FAKE=1 python examples/03_query_and_context.py     (hors ligne)
    python examples/03_query_and_context.py                  (avec ANTHROPIC_API_KEY)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import bench, contextual, llm, query  # noqa: E402

if __name__ == "__main__":
    articles, chunks, embedder, dense, lexical, hybrid = bench.build()
    question = "Quel est le fournisseur du bureau Lift et combien de temps ses moteurs sont-ils garantis ?"

    print("Sous-questions :", query.decompose(question))
    print("Réécriture :", query.rewrite("How do I reset my standing desk?"))
    print("HyDE :", query.hyde("délai de remboursement")[:120], "…")
    print("Multi-query :", [h.chunk.article_id for h in query.search_multi(hybrid, question)])
    print("Décomposée :", [h.chunk.article_id for h in query.search_decomposed(hybrid, question)])

    enriched = contextual.contextualize(chunks[:4], articles)
    print("\nPassage contextualisé :\n", enriched[0].indexed_text())
    parents = contextual.expand_to_parents(hybrid.search("vérin de la chaise", 10), articles, 2)
    print("\nParents :", [p.chunk.article_id for p in parents])
    print("\nAppels de LLM par tâche :", llm.USAGE.calls,
          f"| cache : {llm.USAGE.cache_read_tokens} tokens lus, {llm.USAGE.cache_write_tokens} écrits")
