"""Leçons 4.2 et 4.3 — RAG agentique et RAG auto-correctif.

    PRISM_FAKE=1 python examples/05_agentic_and_corrective.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prism import bench  # noqa: E402
from prism.agentic import AgenticRAG  # noqa: E402
from prism.corrective import corrective_answer  # noqa: E402

if __name__ == "__main__":
    articles, chunks, embedder, dense, lexical, hybrid = bench.build()

    agent = AgenticRAG(hybrid, articles)
    result = agent.run("Quel est le fournisseur du bureau Lift et combien de temps ses moteurs sont garantis ?")
    print("Agent — appels d'outils :", result.tool_calls)
    print("Agent — réponse :", result.answer, f"({result.turns} tours)\n")

    for question in ["Combien coûte la livraison express ?", "Vendez-vous des imprimantes 3D ?"]:
        r = corrective_answer(hybrid, question)
        print(f"CRAG « {question} »\n  actions : {r.actions}\n  réponse : {r.answer}  sources {r.sources}")
