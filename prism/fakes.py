"""Réponses factices et déterministes pour le mode hors ligne (PRISM_FAKE=1).

Chaque appel au LLM porte un nom de tâche : la réponse factice est choisie selon la tâche, jamais en
devinant le prompt. Ce fichier n'est pas étudié dans le cours.
"""

from __future__ import annotations

import json
import re
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from .bm25 import tokenize

KNOWN = {
    "Chaise Axis": "PRODUIT", "Bureau Lift": "PRODUIT", "Lampe Halo": "PRODUIT", "Mug Thermo": "PRODUIT",
    "Roulettes CASTER-M": "PRODUIT", "Ergoform": "FOURNISSEUR", "Lumina": "FOURNISSEUR",
    "Inox et Compagnie": "FOURNISSEUR", "Allemagne": "PAYS", "France": "PAYS", "Portugal": "PAYS",
    "Garantie": "POLITIQUE", "Retours": "POLITIQUE", "Livraison": "POLITIQUE",
}
ALIASES = {"Chaise Axis": ["axis", "chair-7"], "Bureau Lift": ["lift", "desk-2"], "Lampe Halo": ["halo", "lamp-3"],
           "Mug Thermo": ["thermo", "mug-01"], "Roulettes CASTER-M": ["caster-m"], "Ergoform": ["ergoform"],
           "Lumina": ["lumina"], "Inox et Compagnie": ["inox"], "Allemagne": ["allemagne"], "France": ["france"],
           "Portugal": ["portugal"], "Garantie": ["garanti"], "Retours": ["retour", "rétractation"],
           "Livraison": ["livr"]}
RELATIONS = [("Ergoform", "Chaise Axis", "FABRIQUE"), ("Ergoform", "Bureau Lift", "FABRIQUE"),
             ("Lumina", "Lampe Halo", "FABRIQUE"), ("Inox et Compagnie", "Mug Thermo", "FABRIQUE"),
             ("Ergoform", "Allemagne", "SITUE_EN"), ("Lumina", "France", "SITUE_EN"),
             ("Inox et Compagnie", "Portugal", "SITUE_EN"), ("Roulettes CASTER-M", "Chaise Axis", "COMPATIBLE_AVEC"),
             ("Garantie", "Chaise Axis", "COUVRE"), ("Garantie", "Lampe Halo", "COUVRE"), ("Garantie", "Bureau Lift", "COUVRE"),
             ("Retours", "Livraison", "LIE_A")]


def _entities(text: str) -> list[str]:
    lower = text.lower()
    return [name for name, aliases in ALIASES.items() if any(a in lower for a in aliases)]


def _best_sentences(context: str, question: str, n: int = 2) -> str:
    q = set(tokenize(question))
    lines = [l.strip() for l in re.split(r"(?<=[.!?])\s+|\n+", context) if len(l.strip()) > 25]
    ranked = sorted(lines, key=lambda l: -len(q & set(tokenize(l))))
    if not ranked or not q & set(tokenize(ranked[0])):
        return "Je ne trouve pas cette information dans la base."
    return " ".join(ranked[:n])


def complete(task: str, prompt: str, cached_prefix: str | None = None) -> str:
    question = re.findall(r"Question : (.*)", prompt)
    q = question[-1] if question else prompt
    if task == "rewrite":
        return " ".join(tokenize(q))
    if task == "hyde":
        return f"{q} Le centre d'aide précise les conditions, les délais et les frais applicables."
    if task == "contextualize":
        title = re.search(r"# (.*)", cached_prefix or "")
        return f"Ce passage provient de l'article « {title.group(1) if title else 'inconnu'} »."
    if task == "community_summary":
        return "Groupe de fournisseurs, produits, pays et politiques : " + "; ".join(prompt.splitlines()[1:4])
    if task == "global_reduce":
        return " ".join(l[2:] for l in prompt.splitlines() if l.startswith("- "))
    if task == "answer":
        context = prompt.split("Extraits :", 1)[-1].split("Question :", 1)[0]
        best = _best_sentences(context, q, 1)
        if best.startswith("Je ne trouve"):
            return best
        line = next((l for l in context.splitlines() if best[:40] in l), "")
        cited = re.match(r"\[([a-z]+-\d+)\]", line.strip())
        return best + (f" [{cited.group(1)}]" if cited else "")
    return _best_sentences(prompt, q, 1)


def _fill(schema: type[BaseModel], text: str) -> BaseModel:
    """Remplissage générique d'un schéma inconnu (ex. modèles internes de RAGAS) : valeurs plausibles."""
    values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        ann = field.annotation
        origin = get_origin(ann)
        if ann is bool:
            values[name] = True
        elif ann is int:
            values[name] = 1
        elif ann is float:
            values[name] = 1.0
        elif ann is str:
            values[name] = text[:80] or "ok"
        elif origin is list:
            (inner,) = get_args(ann) or (str,)
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                values[name] = [_fill(inner, text)]
            else:
                values[name] = [text[:80] or "ok"] if inner is str else [1]
        elif isinstance(ann, type) and issubclass(ann, BaseModel):
            values[name] = _fill(ann, text)
        elif get_args(ann):
            values[name] = get_args(ann)[0]
        else:
            values[name] = None
    return schema.model_validate(values)


def structured(task: str, schema: type[BaseModel], prompt: str) -> BaseModel:
    q = (re.findall(r"Question : (.*)", prompt) or [prompt])[-1]
    if task == "multi_query":
        words = tokenize(q)
        return schema(queries=[" ".join(words), " ".join(reversed(words)), q.lower()])
    if task == "decompose":
        parts = [p.strip(" ?") + " ?" for p in re.split(r"\s+et\s+|,\s*|\?\s+", q) if len(p.split()) >= 3]
        return schema(questions=parts[:3] or [q])
    if task == "grade":
        passage = prompt.split("Passage : ", 1)[-1]
        overlap = len(set(tokenize(q)) & set(tokenize(passage)))
        return schema(verdict="pertinent" if overlap >= 2 else "partiel" if overlap == 1 else "hors_sujet")
    if task == "support_check":
        return schema(supported=True, unsupported=[])
    if task == "extract_graph":
        text = prompt.split("Passage", 1)[-1]
        names = _entities(text)
        entities = [{"name": n, "type": KNOWN[n]} for n in names]
        rels = [{"source": s, "target": t, "label": l} for s, t, l in RELATIONS if s in names and t in names]
        return schema(entities=entities, relations=rels)
    if task == "global_map":
        summary = prompt.split("Question :")[0]
        score = len(set(tokenize(q)) & set(tokenize(summary)))
        return schema(answer=summary.split(":", 1)[-1].strip()[:200] if score else "", score=min(score * 3, 10))
    if task == "judge_statements":
        text = prompt.split(":", 1)[-1]
        return schema(statements=[s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3])
    if task == "judge_support":
        contexts = prompt.split("Affirmations :")[0]
        statements = re.findall(r"^\d+\. (.*)$", prompt, re.M)
        ctx = set(tokenize(contexts))
        def ok(statement: str) -> bool:  # soutenue si au moins 70 % de ses mots figurent dans les extraits
            words = set(tokenize(statement))
            return len(words & ctx) >= max(1, round(0.7 * len(words)))
        return schema(supported=[ok(s) for s in statements])
    if task == "judge_useful":
        reference = re.search(r"Réponse de référence : (.*)", prompt).group(1)
        passage = prompt.split("Passage : ", 1)[-1]
        return schema(useful=len(set(tokenize(reference)) & set(tokenize(passage))) >= 1)
    if task == "testset_generate":
        title = re.search(r"\((.*?)\)", prompt)
        passage = prompt.split(":\n", 1)[-1].split("\n\nRédige")[0]
        first = re.split(r"(?<=[.!?])\s+", passage)[0]
        subject = (title.group(1) if title else "ce sujet").lower()
        return schema(question=f"J'aurais une question au sujet de : {subject} ?", answer=first, style="reformulee")
    if task == "testset_check":
        return schema(answerable=True)
    return _fill(schema, q)


# --- Client Anthropic factice pour l'agent (même forme de réponses que l'API) -------------------

class FakeAgentClient:
    """Imite client.messages.create : cherche une fois par volet de la question, puis répond."""

    def __init__(self) -> None:
        self.messages = self

    def create(self, **kwargs):
        from anthropic.types import Message

        messages = kwargs["messages"]
        question = messages[0]["content"]
        tool_results = [b for m in messages if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]
        parts = [p for p in re.split(r"\s+et\s+|\?\s+", question) if len(p.split()) >= 3] or [question]
        usage = {"input_tokens": 100, "output_tokens": 20}
        if len(tool_results) < len(parts):
            part = parts[len(tool_results)]
            block = {"type": "tool_use", "id": f"call_{len(tool_results)}", "name": "search_help_center",
                     "input": {"query": " ".join(tokenize(part))}}
            return Message.model_validate({"id": "m", "type": "message", "role": "assistant", "model": "fake",
                                           "content": [block], "stop_reason": "tool_use", "stop_sequence": None,
                                           "usage": usage})
        found = []
        for r in tool_results:
            hits = json.loads(r["content"]) if r["content"].startswith("[") else []
            found += [f"{h['texte']} [{h['article']}]" for h in hits[:2]]
        text = " ".join(found) or "Je ne trouve pas cette information dans la base."
        return Message.model_validate({"id": "m", "type": "message", "role": "assistant", "model": "fake",
                                       "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                                       "stop_sequence": None, "usage": usage})

