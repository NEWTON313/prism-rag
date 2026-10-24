"""Index ANN (leçon 2.1) : comparer exactitude, vitesse et mémoire de FAISS Flat, HNSW, IVF et IVF-PQ.

On mesure toujours un index approché CONTRE la recherche exacte : recall@10 = part des 10 vrais plus
proches voisins retrouvés. Un index rapide mais à 70 % de recall perd silencieusement des documents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def synthetic_vectors(n: int, dim: int = 384, clusters: int = 50, intrinsic: int = 32, seed: int = 0,
                      projection_seed: int = 42) -> np.ndarray:
    """Vecteurs réalistes : des thèmes dans un espace de faible dimension « intrinsèque », plongés en `dim`.

    De vrais embeddings occupent une petite partie de l'espace (une structure de faible dimension). Du bruit
    pur en 384 dimensions serait trompeur : tous les points y sont presque équidistants, et la notion même de
    « plus proche voisin » perd son sens — aucun index approché n'y paraîtrait bon.
    """
    rng = np.random.default_rng(seed)
    projection = np.random.default_rng(projection_seed).normal(size=(intrinsic, dim))  # même espace pour base et requêtes
    centers = np.random.default_rng(projection_seed + 1).normal(size=(clusters, intrinsic))
    labels = rng.integers(0, clusters, size=n)
    latent = centers[labels] + 0.5 * rng.normal(size=(n, intrinsic))
    vectors = latent @ projection + 0.05 * rng.normal(size=(n, dim))
    return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)


@dataclass
class AnnResult:
    name: str
    recall_at_10: float
    ms_per_query: float
    build_seconds: float
    bytes_per_vector: float


def _measure(name: str, index, base: np.ndarray, queries: np.ndarray, truth: np.ndarray, train: bool) -> AnnResult:
    import faiss

    start = time.perf_counter()
    if train:
        index.train(base)
    index.add(base)
    build = time.perf_counter() - start
    start = time.perf_counter()
    _, found = index.search(queries, 10)
    elapsed = (time.perf_counter() - start) * 1000 / len(queries)
    recall = np.mean([len(set(found[i]) & set(truth[i])) / 10 for i in range(len(queries))])
    size = faiss.serialize_index(index).nbytes / len(base)
    return AnnResult(name, float(recall), elapsed, build, size)


def compare_indexes(n: int = 50_000, dim: int = 384, n_queries: int = 200) -> list[AnnResult]:
    import faiss

    base = synthetic_vectors(n, dim)
    queries = synthetic_vectors(n_queries, dim, seed=1)
    flat = faiss.IndexFlatIP(dim)  # produit scalaire exact = cosinus sur vecteurs normalisés
    flat.add(base)
    _, truth = flat.search(queries, 10)

    results = [_measure("Flat (exact)", faiss.IndexFlatIP(dim), base, queries, truth, train=False)]
    for ef in (16, 64, 256):  # efSearch : taille de la liste de candidats explorée par requête
        hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)  # M = 32 voisins par nœud
        hnsw.hnsw.efConstruction = 100
        hnsw.hnsw.efSearch = ef
        results.append(_measure(f"HNSW M=32 efSearch={ef}", hnsw, base, queries, truth, train=False))
    for nprobe in (4, 32):  # nprobe : nombre de listes (sur 256) explorées par requête
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, 256, faiss.METRIC_INNER_PRODUCT)
        ivf.nprobe = nprobe
        results.append(_measure(f"IVF nlist=256 nprobe={nprobe}", ivf, base, queries, truth, train=True))
    ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, 256, 48, 8, faiss.METRIC_INNER_PRODUCT)
    ivfpq.nprobe = 32  # 48 sous-vecteurs codés sur 8 bits : 48 octets par vecteur au lieu de 1 536
    results.append(_measure("IVF-PQ m=48 nprobe=32", ivfpq, base, queries, truth, train=True))
    return results
