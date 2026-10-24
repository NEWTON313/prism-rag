"""Corpus et découpage (module 1, module 3).

Un Chunk garde toujours l'identifiant de son article d'origine : c'est ce qui permet de mesurer la
recherche au niveau de l'article (le découpage peut changer, le jeu d'évaluation reste valable).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Article:
    id: str
    category: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    article_id: str
    category: str
    title: str
    text: str
    parent_id: str | None = None  # pour le « parent-document retrieval » (leçon 3.4)
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def indexed_text(self) -> str:
        """Ce qui est vectorisé et indexé : le titre donne son contexte au passage."""
        context = self.meta.get("context")
        return f"{self.title}\n{context + chr(10) if context else ''}{self.text}"


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    relevant: list[str]
    answer: str | None


def load_articles(path: Path | None = None) -> list[Article]:
    path = path or config.DATA_DIR / "articles.jsonl"
    return [Article(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_questions(path: Path | None = None) -> list[Question]:
    path = path or config.DATA_DIR / "questions.jsonl"
    return [Question(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_articles(articles: list[Article], max_sentences: int = 2) -> list[Chunk]:
    """Découpe chaque article en passages de quelques phrases (petits passages : recherche précise)."""
    chunks = []
    for article in articles:
        sentences = SENTENCE.split(article.text)
        for i in range(0, len(sentences), max_sentences):
            chunks.append(Chunk(
                id=f"{article.id}#{i // max_sentences}", article_id=article.id, category=article.category,
                title=article.title, text=" ".join(sentences[i:i + max_sentences]), parent_id=article.id,
            ))
    return chunks


def whole_articles(articles: list[Article]) -> list[Chunk]:
    """Un passage par article : la granularité « parent »."""
    return [Chunk(id=a.id, article_id=a.id, category=a.category, title=a.title, text=a.text) for a in articles]
