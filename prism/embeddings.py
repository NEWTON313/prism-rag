"""Embeddings (leçon 1.3) : un modèle local réel (FastEmbed) ou un modèle factice déterministe.

Les vecteurs sont NORMALISÉS : le produit scalaire est alors égal à la similarité cosinus.
Un cache disque évite de recalculer les embeddings d'un banc d'essai à l'autre.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class FastEmbedEmbedder:
    def __init__(self, model: str = config.EMBED_MODEL) -> None:
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model, cache_dir=str(config.CACHE_DIR / "models"))
        self.dim = len(next(iter(self._model.embed(["test"]))))
        self._cache_file = config.CACHE_DIR / f"emb-{hashlib.md5(model.encode()).hexdigest()[:10]}.npz"
        self._cache: dict[str, np.ndarray] = {}
        if self._cache_file.exists():
            data = np.load(self._cache_file)
            self._cache = {k: data[k] for k in data.files}

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self._cache]
        if missing:
            for text, vector in zip(missing, self._model.embed(missing)):
                self._cache[hashlib.md5(text.encode()).hexdigest()] = np.asarray(vector, dtype=np.float32)
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(self._cache_file, **self._cache)
        return normalize(np.stack([self._cache[k] for k in keys]))


WORD = re.compile(r"[a-zà-ÿ0-9]+(?:-[a-z0-9]+)?", re.I)


class HashEmbedder:
    """Factice : sac de racines de mots haché. Assez « sémantique » pour des tests, rien de plus."""

    def __init__(self, dim: int = 256) -> None:
        self.name, self.dim = "hash", dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in WORD.findall(text.lower()):
                if len(word) > 2:
                    out[i, int(hashlib.md5(word[:6].encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalize(out + 1e-6)


@lru_cache(maxsize=4)
def default_embedder() -> Embedder:
    return HashEmbedder() if config.FAKE else FastEmbedEmbedder()
