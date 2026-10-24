# Prism — le laboratoire RAG avancé

> Projet compagnon du cours gratuit **Advanced RAG et bases vectorielles** — https://iamaestro.academy/fr/courses/advanced-rag

Projet compagnon du cours IAMaestro **« Advanced RAG & Bases de Données Vectorielles »**.

Prism implémente, en Python lisible, chaque technique du cours — et les MESURE sur un même jeu de données :
le centre d'aide de la boutique fictive *Nimbus* (36 articles, 67 passages) et 45 questions annotées
(factuelles, reformulées, lexicales, en anglais, à plusieurs sources, hors base).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -e ".[dev,ann]"        # ajoutez « ragas » pour la bibliothèque RAGAS
```

Python 3.11 ou plus récent. Les embeddings et le re-ranker tournent **localement** (FastEmbed) : aucun coût,
aucune clé ; le premier lancement télécharge environ 1,3 Go de modèles.

## Exécuter

```bash
python examples/01_benchmark.py          # le banc d'essai complet, modèles réels
python examples/02_ann_indexes.py        # FAISS : Flat, HNSW, IVF, IVF-PQ
PRISM_FAKE=1 pytest                      # 25 tests hors ligne (+1 si ragas est installé)
```

Les techniques à base de LLM (réécriture, HyDE, contextual retrieval, GraphRAG, agent, CRAG, juge) utilisent
Claude : définissez `ANTHROPIC_API_KEY`. Avec `PRISM_FAKE=1`, un LLM factice déterministe les exécute hors ligne
(la mécanique est vérifiée, les résultats n'ont pas de valeur).

## Résultats mesurés (modèles réels, CPU)

| Configuration | Recall@5 | MRR | Anglais | Reformulées | Multi-sources |
|---|---|---|---|---|---|
| BM25 | 0,74 | 0,70 | 0,00 | 0,55 | 1,00 |
| Dense | 0,90 | 0,82 | 0,92 | 0,90 | 0,75 |
| Hybride (RRF) | 0,99 | 0,82 | 0,92 | 1,00 | 1,00 |
| Dense binaire | 0,84 | 0,78 | 1,00 | 0,80 | 0,42 |
| Dense binaire + re-scoring | 0,90 | 0,82 | 0,92 | 0,90 | 0,75 |
| Hybride + re-ranking | 1,00 | 0,97 | 1,00 | 1,00 | 1,00 |

| Index (50 000 vecteurs, 384 dim.) | recall@10 | ms/requête | octets/vecteur |
|---|---|---|---|
| Flat (exact) | 1,000 | 0,58 | 1 536 |
| HNSW M=32, efSearch=64 | 1,000 | 0,03 | 1 808 |
| IVF nlist=256, nprobe=4 | 0,932 | 0,03 | 1 552 |
| IVF-PQ m=48, nprobe=32 | 0,593 | 0,03 | 72 |

## Contenu

| Fichier | Rôle | Leçons |
|---|---|---|
| `prism/corpus.py`, `data/` | Corpus, découpage, questions annotées | 1.1, 1.2 |
| `prism/embeddings.py` | Embeddings réels (FastEmbed) ou factices, cache disque | 1.3 |
| `prism/metrics.py`, `prism/bench.py` | Recall@k, MRR, nDCG, banc d'essai | 1.2, 1.4 |
| `prism/ann.py` | Comparaison d'index ANN avec FAISS | 2.1 |
| `sql/` | pgvector : HNSW, filtres, RLS multi-tenant, halfvec, binaire, hybride SQL | 2.2, 2.3, 2.4, 3.1 |
| `prism/vector.py` | Index exact, quantization int8 et binaire, re-scoring, filtres | 2.3, 2.4 |
| `prism/bm25.py`, `prism/retrieval.py` | BM25, dense, hybride, RRF, fusion pondérée | 3.1 |
| `prism/rerank.py` | Cross-encoder multilingue | 3.2 |
| `prism/query.py` | Réécriture, HyDE, multi-query, décomposition | 3.3 |
| `prism/contextual.py` | Contextual retrieval (avec prompt caching), parent-document | 3.4 |
| `prism/graphrag.py` | Graphe d'entités, communautés de Louvain, recherches locale et globale | 4.1 |
| `prism/agentic.py` | RAG agentique (boucle d'outils Claude) | 4.2 |
| `prism/corrective.py`, `prism/generation.py` | CRAG, vérification à la Self-RAG, citations | 4.3 |
| `prism/multimodal.py` | Images décrites par la vision, tableaux linéarisés | 4.4 |
| `prism/judge.py`, `prism/ragas_adapter.py` | Faithfulness, context precision/recall ; RAGAS sur Claude | 5.1 |
| `prism/testset.py` | Jeux de tests synthétiques filtrés | 5.2 |
| `prism/cache.py`, `prism/llm.py` | Cache sémantique, suivi des tokens et du cache de prompt | 5.3 |
| `prism/telemetry.py` | Traces et revue hebdomadaire | 5.4 |

## Licence

Le code est distribué sous licence MIT (voir `LICENSE`) : vous pouvez l'utiliser, le modifier et le redistribuer, y compris dans des projets professionnels. Les textes du cours restent protégés par le droit d'auteur. Les modèles téléchargés restent soumis à leurs propres licences.

Le cours complet, gratuit, en français et en anglais : https://iamaestro.academy/fr/courses/advanced-rag
