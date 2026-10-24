"""Leçon 4.1 — GraphRAG : graphe d'entités, communautés, recherche locale et globale.

    PRISM_FAKE=1 python examples/04_graphrag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import bench, graphrag  # noqa: E402

if __name__ == "__main__":
    articles, chunks, *_ = bench.build()
    kg = graphrag.build_graph(chunks)
    print(f"Graphe : {kg.graph.number_of_nodes()} entités, {kg.graph.number_of_edges()} relations, "
          f"{len(kg.communities)} communautés")
    for community in kg.communities:
        print(f"  communauté {community.id} : {', '.join(community.entities)}")

    by_id = {c.id: c for c in chunks}
    local = graphrag.local_search(kg, "Qui fabrique la chaise Axis ?", by_id)
    print("\nRecherche locale :", [c.id for c in local])
    print("Recherche globale :", graphrag.global_search(kg, "Quels sont les fournisseurs et leurs pays ?"))
