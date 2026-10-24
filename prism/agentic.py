"""RAG agentique (leçon 4.2) : le modèle décide QUAND et COMMENT chercher.

Au lieu d'une recherche unique avant la réponse, Claude dispose d'outils de recherche, choisit ses requêtes,
lit les résultats, cherche encore si besoin (autre formulation, autre catégorie, article complet), puis répond.
Boucle d'agent classique, bornée en nombre de tours.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config, llm
from .corpus import Article

TOOLS = [
    {
        "name": "search_help_center",
        "description": "Recherche hybride (sens + mots-clés) dans le centre d'aide Nimbus. "
                       "Renvoie au plus 5 passages avec l'identifiant de leur article. "
                       "Formule une requête courte et précise ; relance avec d'autres mots "
                       "si les résultats ne répondent pas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Requête de recherche, en français"},
                "category": {"type": "string", "enum": ["retours", "garantie", "livraison", "compte", "paiement",
                                                        "produits", "entreprises", "fidelite", "environnement",
                                                        "entreprise"],
                             "description": "Optionnel : restreint la recherche à une catégorie"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_article",
        "description": "Renvoie le texte complet d'un article à partir de son identifiant (ex. gar-05), "
                       "quand un passage ne suffit pas.",
        "input_schema": {
            "type": "object",
            "properties": {"article_id": {"type": "string"}},
            "required": ["article_id"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = """Tu es l'assistant du support de la boutique Nimbus. Utilise les outils pour trouver
l'information avant de répondre : ne réponds jamais de mémoire, car les règles de la boutique te
sont inconnues. Si une question a plusieurs volets, cherche chacun. Cite les identifiants d'articles
utilisés entre crochets. Si l'information est introuvable après deux ou trois recherches, dis-le
simplement."""


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    turns: int = 0


class AgenticRAG:
    def __init__(self, retriever, articles: list[Article], max_turns: int = 6) -> None:
        self.retriever, self.max_turns = retriever, max_turns
        self.articles = {a.id: a for a in articles}

    def _run_tool(self, name: str, args: dict) -> str:
        if name == "search_help_center":
            category = args.get("category")
            where = (lambda c: c.category == category) if category else None
            hits = self.retriever.search(args["query"], 5, where)
            return json.dumps([{"article": h.chunk.article_id, "titre": h.chunk.title, "texte": h.chunk.text}
                               for h in hits], ensure_ascii=False) or "[]"
        if name == "read_article":
            article = self.articles.get(args["article_id"])
            return f"# {article.title}\n{article.text}" if article else f"Article inconnu : {args['article_id']}"
        return f"Outil inconnu : {name}"

    def run(self, question: str, client=None) -> AgentResult:
        client = client or llm.client()
        messages: list[dict] = [{"role": "user", "content": question}]
        result = AgentResult(answer="")
        for turn in range(1, self.max_turns + 1):
            response = client.messages.create(model=config.MODEL, max_tokens=4096, system=SYSTEM,
                                              tools=TOOLS, messages=messages)
            llm.USAGE.add("agent", response.usage)
            # La réponse complète (réflexion, texte, appels d'outils) retourne dans l'historique
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                result.answer = "".join(b.text for b in response.content if b.type == "text").strip()
                result.turns = turn
                return result
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    result.tool_calls.append((block.name, dict(block.input)))
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": self._run_tool(block.name, block.input)})
            messages.append({"role": "user", "content": results})
        result.answer = "Je n'ai pas pu terminer la recherche ; un conseiller va prendre le relais."
        result.turns = self.max_turns
        return result
