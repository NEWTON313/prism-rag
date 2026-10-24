"""Leçons 1.2, 2.4, 3.1 et 3.2 — Le banc d'essai complet : BM25, dense, hybride, quantization, re-ranking.

    python examples/01_benchmark.py                 (modèles réels, locaux : premier lancement ≈ 1,3 Go)
    PRISM_FAKE=1 python examples/01_benchmark.py    (hors ligne : vérifie la mécanique, chiffres sans valeur)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import bench  # noqa: E402

if __name__ == "__main__":
    results = bench.run(with_rerank="--no-rerank" not in sys.argv)
    bench.print_table(results)
    worst = min(results, key=lambda r: r[0].recall)[0]
    print(f"\nÉchecs de « {worst.name} » :")
    for failure in worst.failures:
        print("  -", failure)
