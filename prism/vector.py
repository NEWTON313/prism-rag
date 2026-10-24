"""Index vectoriel exact et quantization (leçons 2.3 et 2.4).

Recherche exacte par produit scalaire (vecteurs normalisés) : la référence à laquelle on compare les index
approchés (module 2). Trois représentations des vecteurs :

    float32 : 4 octets par dimension — la référence
    int8    : 1 octet par dimension (×4 moins de mémoire), perte de précision faible
    binary  : 1 bit par dimension (×32 moins de mémoire), à compléter par un re-scoring en float32
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .corpus import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, quantization: str = "float32") -> None:
        if quantization not in ("float32", "int8", "binary"):
            raise ValueError(quantization)
        self.chunks, self.quantization = chunks, quantization
        self.float_vectors = vectors.astype(np.float32)
        if quantization == "int8":
            # Quantization scalaire : chaque dimension est ramenée sur [-127, 127] avec son propre facteur
            self.scale = np.abs(vectors).max(axis=0) / 127.0 + 1e-12
            self.codes = np.round(vectors / self.scale).astype(np.int8)
        elif quantization == "binary":
            self.codes = np.packbits(vectors > 0, axis=1)  # un bit par dimension : le signe

    def memory_bytes(self) -> int:
        """Mémoire des vecteurs utilisés pour la recherche (hors re-scoring)."""
        return self.float_vectors.nbytes if self.quantization == "float32" else self.codes.nbytes

    def _raw_scores(self, query: np.ndarray) -> np.ndarray:
        if self.quantization == "float32":
            return self.float_vectors @ query
        if self.quantization == "int8":
            return (self.codes.astype(np.float32) * self.scale) @ query
        # binaire : similarité = nombre de bits identiques (distance de Hamming inversée)
        q_bits = np.packbits(query > 0)
        differing = np.unpackbits(np.bitwise_xor(self.codes, q_bits), axis=1).sum(axis=1)
        return -differing.astype(np.float32)

    def search(self, query: np.ndarray, k: int = 5, where: Callable[[Chunk], bool] | None = None,
               rescore: int = 0) -> list[Hit]:
        """`where` filtre AVANT le classement ; `rescore` re-classe en float32 les N meilleurs candidats."""
        scores = self._raw_scores(query)
        if where is not None:
            mask = np.array([where(c) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        candidates = np.argsort(-scores)[: max(k, rescore)]
        candidates = [i for i in candidates if np.isfinite(scores[i])]
        if rescore:
            exact = self.float_vectors[candidates] @ query
            order = np.argsort(-exact)[:k]
            return [Hit(self.chunks[candidates[i]], float(exact[i])) for i in order]
        return [Hit(self.chunks[i], float(scores[i])) for i in candidates[:k]]
