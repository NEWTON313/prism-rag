"""Techniques à base de LLM, en mode hors ligne : requêtes, contexte, GraphRAG, agent, CRAG, multimodal,
évaluation, jeux de tests, cache, traces. Modules 3 à 5."""

from pathlib import Path

from prism import cache, config, contextual, graphrag, judge, llm, multimodal, query, telemetry, testset
from prism.agentic import AgenticRAG
from prism.corrective import corrective_answer
from prism.embeddings import default_embedder
from prism.generation import answer_with_citations


def test_query_transformations(hybrid, dense):
    assert query.decompose("Quel est le fournisseur du bureau et combien de temps ses moteurs sont garantis ?") != []
    hits = query.search_multi(hybrid, "Combien coûte la livraison express ?")
    assert "liv-02" in {h.chunk.article_id for h in hits}
    assert query.search_hyde(dense, "délai de remboursement")  # renvoie des résultats


def test_contextual_retrieval_adds_context(chunks, articles):
    contextualized = contextual.contextualize(chunks[:3], articles)
    assert all("Ce passage provient de l'article" in c.indexed_text() for c in contextualized)
    assert chunks[0].meta == {}  # les passages d'origine ne sont pas modifiés


def test_parent_document_expansion(hybrid, articles):
    parents = contextual.expand_to_parents(hybrid.search("vérin garantie chaise", 10), articles, 3)
    assert len({p.chunk.article_id for p in parents}) == len(parents)  # pas de doublon d'article
    assert all(p.chunk.text in {a.text for a in articles} for p in parents)  # l'article entier


def test_graphrag(chunks):
    kg = graphrag.build_graph(chunks)
    assert kg.graph.has_edge("Ergoform", "Chaise Axis")
    assert kg.communities and all(c.summary for c in kg.communities)
    by_id = {c.id: c for c in chunks}
    local = graphrag.local_search(kg, "Qui fabrique la chaise Axis ?", by_id)
    assert "fourn-01" in {c.article_id for c in local}
    assert graphrag.global_search(kg, "Quels fournisseurs et quels pays ?")


def test_agentic_rag_searches_each_part(hybrid, articles):
    result = AgenticRAG(hybrid, articles).run("Quel est le fournisseur du bureau Lift et combien de temps ses moteurs sont garantis ?")
    assert [name for name, _ in result.tool_calls] == ["search_help_center", "search_help_center"]
    assert "Ergoform" in result.answer and result.turns == 3


def test_corrective_rag(hybrid):
    ok = corrective_answer(hybrid, "Combien coûte la livraison express ?")
    assert "liv-02" in ok.sources and "9,90" in ok.answer
    missing = corrective_answer(hybrid, "Recette du gâteau au chocolat ?")
    assert missing.sources == [] and any(a.startswith("reformulation") for a in missing.actions)


def test_answer_cites_only_provided_articles(hybrid):
    answer, cited = answer_with_citations("Combien coûte la livraison express ?", hybrid.search("livraison express", 3))
    assert cited == ["liv-02"] and "[liv-02]" in answer


def test_multimodal_and_tables():
    chunks = multimodal.image_chunks(config.DATA_DIR / "images", {"desk-2-schema.png": "desk-01", "tarifs-livraison.png": "liv-02"})
    assert {c.article_id for c in chunks} == {"desk-01", "liv-02"}
    assert "M6" in next(c.text for c in chunks if c.article_id == "desk-01")
    text = multimodal.table_to_text(["Mode", "Prix"], [["Express", "9,90 €"]], "Tarifs")
    assert "Mode : Express ; Prix : 9,90 €." in text


def test_judge_metrics():
    contexts = ["La livraison express coûte 9,90 euros."]
    assert judge.faithfulness("La livraison express coûte 9,90 euros.", contexts) == 1.0
    assert judge.faithfulness("La livraison express est gratuite le dimanche à Lyon.", contexts) < 1.0
    assert judge.context_recall("9,90 euros pour l'express.", contexts) == 1.0
    assert judge.context_precision("prix express ?", "9,90 euros", ["Sans rapport.", contexts[0]]) == 0.5


def test_synthetic_testset(chunks):
    cases, rejected = testset.generate_testset(chunks[:10], default_embedder())
    assert cases and set(rejected) == {"non_repondable", "trop_recopiee", "doublon"}
    assert all(testset.lexical_overlap(c.question, next(x.text for x in chunks if x.id == c.source_chunk)) <= 0.6
               for c in cases)


def test_semantic_cache():
    c = cache.SemanticCache(default_embedder(), threshold=0.9)
    c.store("Combien coûte la livraison express ?", "9,90 €", ["liv-02"], scope="fr")
    assert c.lookup("combien coûte la livraison express", scope="fr") is not None
    assert c.lookup("Combien coûte la livraison express ?", scope="en") is None  # autre périmètre
    assert c.lookup("Comment supprimer mon compte ?", scope="fr") is None
    assert c.invalidate_sources({"liv-02"}) == 1 and c.stats.hits == 1


def test_traces_and_weekly_review(tmp_path: Path):
    log = telemetry.TraceLog(tmp_path / "traces.jsonl")
    t1 = telemetry.Trace("Prix express ?", [("liv-02#0", 0.8)], "9,90 €", ["liv-02"], feedback=1)
    with t1.step("retrieval"):
        pass
    log.write(t1)
    log.write(telemetry.Trace("Imprimantes 3D ?", [], "Je ne trouve pas.", [], feedback=-1))
    review = telemetry.weekly_review(log.read())
    assert review["requetes"] == 2 and review["sans_reponse"] == ["Imprimantes 3D ?"]
    assert review["taux_satisfaction"] == 0.5


def test_usage_is_tracked():
    assert llm.USAGE.calls.get("grade", 0) > 0
