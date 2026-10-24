"""Faire tourner les métriques de la bibliothèque RAGAS sur Claude (leçon 5.1).

Deux écueils, vérifiés avec ragas 0.4.3 :
1. ragas importe un module retiré de langchain-community 0.4 : installez `langchain-community<0.4`.
2. Son connecteur Anthropic envoie temperature=0.01 ET top_p=0.1 : Claude Opus 5.5 refuse toute temperature,
   Claude Haiku 5.5 n'accepte que 1, et aucun des deux n'accepte les deux paramètres ensemble.
D'où cet adaptateur : RAGAS demande un objet Pydantic, nous le produisons avec les sorties structurées
natives de Claude (prism.llm.structured), sans aucun paramètre d'échantillonnage.
"""

from __future__ import annotations

import asyncio
from typing import TypeVar

from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM

from . import llm

T = TypeVar("T", bound=BaseModel)


class ClaudeRagasLLM(InstructorBaseRagasLLM):
    def __init__(self, fast: bool = False) -> None:
        # le juge doit être au moins aussi capable que le modèle évalué : grand modèle par défaut
        self.fast = fast

    def generate(self, prompt: str, response_model: type[T]) -> T:
        return llm.structured("ragas", response_model, prompt, fast=self.fast)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return await asyncio.to_thread(self.generate, prompt, response_model)
