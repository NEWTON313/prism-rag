"""Point d'accès unique à Claude pour toutes les techniques à base de LLM.

Chaque appel porte un nom de TÂCHE (« hyde », « grade »…). En mode réel, la tâche ne sert qu'aux traces ;
en mode hors ligne (PRISM_FAKE=1), elle sélectionne une réponse factice déterministe (prism/fakes.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    """Tokens consommés, cumulés par tâche : la base du suivi des coûts (leçon 5.3)."""

    calls: dict[str, int] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, task: str, usage) -> None:
        self.calls[task] = self.calls.get(task, 0) + 1
        if usage is not None:
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0


USAGE = Usage()


@lru_cache(maxsize=1)
def client():
    if config.FAKE:
        from .fakes import FakeAgentClient

        return FakeAgentClient()
    import anthropic

    return anthropic.Anthropic(max_retries=3)  # lit ANTHROPIC_API_KEY ; le SDK réessaie 429, 529 et erreurs réseau


def _text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text").strip()


def complete(task: str, prompt: str, system: str | None = None, fast: bool = False, max_tokens: int = 2048,
             cached_prefix: str | None = None) -> str:
    """Appel texte. `cached_prefix` est placé en tête du message et mis en cache (leçon 3.4)."""
    if config.FAKE:
        from . import fakes

        USAGE.add(task, None)
        return fakes.complete(task, prompt, cached_prefix)
    content: list[dict] = []
    if cached_prefix:
        content.append({"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}})
    content.append({"type": "text", "text": prompt})
    response = client().messages.create(
        model=config.FAST_MODEL if fast else config.MODEL,
        max_tokens=max_tokens,
        system=system or "Tu es un assistant précis. Réponds en français.",
        messages=[{"role": "user", "content": content}],
    )
    USAGE.add(task, response.usage)
    return _text(response)


def structured(task: str, schema: type[T], prompt: str, system: str | None = None, fast: bool = True) -> T:
    """Sortie structurée native : la réponse est un objet Pydantic validé."""
    if config.FAKE:
        from . import fakes

        USAGE.add(task, None)
        return fakes.structured(task, schema, prompt)
    response = client().messages.parse(
        model=config.FAST_MODEL if fast else config.MODEL,
        max_tokens=4096,
        system=system or "Tu es un assistant précis.",
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
    )
    USAGE.add(task, response.usage)
    return response.parsed_output
