"""Réglages de Prism, le laboratoire RAG du cours. Tout ce qui se compare dans les bancs d'essai est ici."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PRISM_DATA_DIR", ROOT / "data"))
CACHE_DIR = Path(os.environ.get("PRISM_CACHE_DIR", ROOT / ".cache"))

# Mode hors ligne : embeddings, re-ranker et LLM factices et déterministes (tests, CI)
FAKE = os.environ.get("PRISM_FAKE") == "1"

# Modèles locaux (FastEmbed, sans clé API) — multilingues, car le corpus est en français
EMBED_MODEL = os.environ.get("PRISM_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
RERANK_MODEL = os.environ.get("PRISM_RERANK_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

# Modèles Claude pour les techniques à base de LLM (réécriture, contexte, graphe, agent, évaluation)
MODEL = os.environ.get("PRISM_MODEL", "claude-opus-5-5")
FAST_MODEL = os.environ.get("PRISM_FAST_MODEL", "claude-haiku-5-5")

TOP_K = 5
