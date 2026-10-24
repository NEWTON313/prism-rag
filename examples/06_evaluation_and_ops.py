"""Leçons 5.1 à 5.4 — Juge LLM, jeu de tests synthétique, cache sémantique, traces et revue hebdomadaire.

    PRISM_FAKE=1 python examples/06_evaluation_and_ops.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import bench, cache, judge, telemetry, testset  # noqa: E402
from prism.generation import answer_with_citations  # noqa: E402

if __name__ == "__main__":
    articles, chunks, embedder, dense, lexical, hybrid = bench.build()

    question, reference = "Combien coûte la livraison express ?", "9,90 euros quel que soit le montant."
    hits = hybrid.search(question, 5)
    contexts = [h.chunk.text for h in hits]
    answer, cited = answer_with_citations(question, hits)
    print("Réponse :", answer)
    print(f"faithfulness={judge.faithfulness(answer, contexts):.2f}  "
          f"context_recall={judge.context_recall(reference, contexts):.2f}  "
          f"context_precision={judge.context_precision(question, reference, contexts):.2f}")

    cases, rejected = testset.generate_testset(chunks[:20], embedder)
    print(f"\nJeu synthétique : {len(cases)} questions gardées, rejetées : {rejected}")

    semantic = cache.SemanticCache(embedder, threshold=0.92)
    log = telemetry.TraceLog(Path(tempfile.mkdtemp()) / "traces.jsonl")
    for q in ["Combien coûte la livraison express ?", "combien coûte la livraison express", "Vendez-vous des imprimantes 3D ?"]:
        trace = telemetry.Trace(q)
        with trace.step("cache"):
            hit = semantic.lookup(q)
        if hit:
            trace.answer, trace.cited, trace.cache_hit = hit.answer, hit.sources, True
        else:
            with trace.step("retrieval"):
                found = hybrid.search(q, 5)
            with trace.step("generation"):
                trace.answer, trace.cited = answer_with_citations(q, found)
            if trace.cited:
                semantic.store(q, trace.answer, trace.cited)
        trace.retrieved = [(h.chunk.id, round(h.score, 3)) for h in hybrid.search(q, 3)]
        log.write(trace)
    print("\nRevue hebdomadaire :", telemetry.weekly_review(log.read()))
