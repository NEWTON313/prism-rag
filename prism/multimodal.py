"""RAG multimodal (leçon 4.4) : rendre cherchables les images, schémas et tableaux.

Stratégie « tout en texte » : un modèle de vision DÉCRIT chaque image (contenu, valeurs, libellés), la
description est indexée comme un passage, avec un lien vers l'image. À la réponse, l'image elle-même
peut être renvoyée au modèle pour qu'il la lise directement.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from . import config, llm
from .corpus import Chunk

DESCRIBE = """Décris cette image d'un centre d'aide pour qu'elle soit retrouvée par une recherche
textuelle : type d'image (schéma, tableau, photo, capture), éléments visibles, TOUS les libellés,
valeurs et codes lisibles, et ce qu'elle permet de comprendre. Pas d'interprétation au-delà de ce
qui est visible."""


def describe_image(path: Path) -> str:
    """Description par Claude (vision). En mode hors ligne : description de référence stockée à côté."""
    if config.FAKE:
        return json.loads((path.parent / "descriptions.json").read_text(encoding="utf-8"))[path.name]
    data = base64.standard_b64encode(path.read_bytes()).decode()
    response = llm.client().messages.create(
        model=config.MODEL, max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": DESCRIBE},
        ]}],
    )
    llm.USAGE.add("describe_image", response.usage)
    return "".join(b.text for b in response.content if b.type == "text").strip()


def image_chunks(image_dir: Path, article_for: dict[str, str]) -> list[Chunk]:
    """Un passage par image ; `article_for` rattache chaque image à son article (pour les métriques)."""
    chunks = []
    for path in sorted(image_dir.glob("*.png")):
        chunks.append(Chunk(
            id=f"img:{path.stem}", article_id=article_for.get(path.name, path.stem), category="images",
            title=f"Image : {path.stem}", text=describe_image(path),
            meta={"image_path": str(path), "modality": "image"},
        ))
    return chunks


def table_to_text(headers: list[str], rows: list[list[str]], caption: str) -> str:
    """Linéarise un tableau : une phrase par ligne, avec les en-têtes répétés.

    Chaque ligne reste compréhensible seule, même si le passage est découpé.
    """
    lines = [caption]
    for row in rows:
        lines.append(" ; ".join(f"{h} : {v}" for h, v in zip(headers, row)) + ".")
    return "\n".join(lines)
