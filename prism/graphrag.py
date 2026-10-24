"""GraphRAG (leçon 4.1) : graphe d'entités, communautés, résumés de communautés.

    indexation : passages ─(LLM)─► entités et relations ─► graphe ─(Louvain)─► communautés ─(LLM)─► résumés
    recherche locale  : entités de la question ─► voisinage dans le graphe ─► passages qui les mentionnent
    recherche globale : chaque résumé de communauté répond partiellement (map) ─► synthèse (reduce)

La recherche globale répond aux questions « sur tout le corpus » (« quels sont les grands thèmes ? »),
qu'aucune recherche de passages ne sait traiter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
from pydantic import BaseModel, Field

from . import llm
from .corpus import Chunk


class Entity(BaseModel):
    name: str = Field(description="Nom canonique : « Chaise Axis » plutôt que « la chaise »")
    type: str = Field(description="PRODUIT, FOURNISSEUR, PAYS, POLITIQUE, SERVICE ou AUTRE")


class Relation(BaseModel):
    source: str
    target: str
    label: str = Field(description="Verbe court en majuscules : FABRIQUE, COMPATIBLE_AVEC, COUVRE…")


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class Partial(BaseModel):
    answer: str = Field(description="Réponse partielle à partir de ce résumé, ou chaîne vide si non pertinent")
    score: int = Field(description="Utilité de cette réponse partielle pour la question, de 0 à 10")


EXTRACT = """Extrais les entités (produits, fournisseurs, pays, politiques, services) et leurs relations
de ce passage du centre d'aide. N'invente rien qui n'y figure pas.

Passage ({chunk_id}) : {text}"""


@dataclass
class Community:
    id: int
    entities: list[str]
    summary: str = ""


@dataclass
class KnowledgeGraph:
    graph: nx.Graph = field(default_factory=nx.Graph)
    mentions: dict[str, set[str]] = field(default_factory=dict)  # entité → passages qui la citent
    communities: list[Community] = field(default_factory=list)


def build_graph(chunks: list[Chunk]) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for chunk in chunks:
        prompt = EXTRACT.format(chunk_id=chunk.id, text=chunk.text)
        extraction = llm.structured("extract_graph", Extraction, prompt)
        for entity in extraction.entities:
            kg.graph.add_node(entity.name, type=entity.type)
            kg.mentions.setdefault(entity.name, set()).add(chunk.id)
        for rel in extraction.relations:
            if rel.source in kg.graph and rel.target in kg.graph:
                kg.graph.add_edge(rel.source, rel.target, label=rel.label)
    # Louvain : regroupe les entités densément reliées entre elles. Graine fixe = résultats reproductibles.
    groups = nx.community.louvain_communities(kg.graph, seed=42) if kg.graph.number_of_edges() else []
    kg.communities = [Community(i, sorted(g)) for i, g in enumerate(groups) if len(g) > 1]
    for community in kg.communities:
        edges = [f"{u} —{d.get('label', 'LIÉ_À')}→ {v}" for u, v, d in kg.graph.subgraph(community.entities).edges(data=True)]
        community.summary = llm.complete(
            "community_summary",
            "Résume en deux phrases ce que ce groupe d'entités et de relations décrit :\n" + "\n".join(edges),
            fast=True, max_tokens=200,
        )
    return kg


def local_search(kg: KnowledgeGraph, question: str, chunks_by_id: dict[str, Chunk], k: int = 5) -> list[Chunk]:
    """Entités citées dans la question, puis leurs voisins directs, puis les passages qui les mentionnent."""
    lowered = question.lower()
    seeds = [e for e in kg.graph.nodes if e.lower() in lowered or e.lower().split()[-1] in lowered]
    scope = set(seeds) | {n for s in seeds for n in kg.graph.neighbors(s)}
    ranked = sorted(scope, key=lambda e: (e not in seeds, e))  # les entités de la question d'abord
    chunk_ids: list[str] = []
    for entity in ranked:
        for cid in sorted(kg.mentions.get(entity, ())):
            if cid not in chunk_ids:
                chunk_ids.append(cid)
    return [chunks_by_id[c] for c in chunk_ids[:k]]


def global_search(kg: KnowledgeGraph, question: str, top: int = 3) -> str:
    """Map-reduce sur les résumés de communautés : chaque communauté répond, on synthétise les meilleures."""
    partials = []
    for community in kg.communities:
        prompt = f"Résumé d'un groupe d'informations :\n{community.summary}\n\nQuestion : {question}"
        p = llm.structured("global_map", Partial, prompt)
        if p.answer and p.score > 0:
            partials.append(p)
    best = sorted(partials, key=lambda p: p.score, reverse=True)[:top]
    if not best:
        return "Je ne trouve pas cette information dans la base."
    return llm.complete("global_reduce", "Synthétise ces éléments en une réponse à la question.\n"
                        + "\n".join(f"- {p.answer}" for p in best) + f"\n\nQuestion : {question}")
