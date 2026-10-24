"""Leçon 2.1 — Index ANN : exactitude, vitesse et mémoire de Flat, HNSW, IVF et IVF-PQ (FAISS).

    pip install -e ".[ann]"
    python examples/02_ann_indexes.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism.ann import compare_indexes  # noqa: E402

if __name__ == "__main__":
    print(f"{'Index':<28}{'recall@10':>10}{'ms/requête':>12}{'construction':>14}{'octets/vecteur':>16}")
    for r in compare_indexes():
        print(f"{r.name:<28}{r.recall_at_10:>10.3f}{r.ms_per_query:>12.3f}{r.build_seconds:>13.1f}s{r.bytes_per_vector:>16.0f}")
