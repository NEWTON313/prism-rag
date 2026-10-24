"""Les métriques de la bibliothèque RAGAS, branchées sur Claude via notre adaptateur (leçon 5.1).

Ignoré si ragas n'est pas installé (pip install -e ".[ragas]"). En mode hors ligne, le « juge » est factice :
le test vérifie l'intégration (RAGAS appelle bien l'adaptateur et calcule un score), pas la qualité.
"""

import asyncio

import pytest

pytest.importorskip("ragas")

from ragas.metrics.collections import ContextRecall, Faithfulness  # noqa: E402

from prism import llm  # noqa: E402
from prism.ragas_adapter import ClaudeRagasLLM  # noqa: E402


def test_ragas_metrics_run_through_the_adapter():
    judge = ClaudeRagasLLM()
    before = llm.USAGE.calls.get("ragas", 0)
    faithfulness = asyncio.run(Faithfulness(llm=judge).ascore(
        user_input="Combien coûte la livraison express ?",
        response="La livraison express coûte 9,90 euros.",
        retrieved_contexts=["La livraison express coûte 9,90 euros quel que soit le montant."],
    ))
    recall = asyncio.run(ContextRecall(llm=judge).ascore(
        user_input="Combien coûte la livraison express ?",
        retrieved_contexts=["La livraison express coûte 9,90 euros quel que soit le montant."],
        reference="9,90 euros.",
    ))
    assert 0.0 <= faithfulness.value <= 1.0 and 0.0 <= recall.value <= 1.0
    assert llm.USAGE.calls["ragas"] > before  # RAGAS est bien passé par Claude (ici, le juge factice)
