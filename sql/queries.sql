-- Prism — requêtes de recherche pgvector (leçons 2.2 à 2.4 et 3.1). $1 = embedding de la question.

-- name: dense
-- Recherche dense filtrée : <=> est la distance cosinus (1 − similarité).
SET hnsw.ef_search = 100;              -- candidats explorés : plus = meilleur rappel, plus lent
-- pgvector ≥ 0.8 : continue l'exploration si le filtre élimine trop de candidats
SET hnsw.iterative_scan = relaxed_order;
SELECT id, article_id, 1 - (embedding <=> $1) AS similarity
FROM chunks
WHERE category = 'garantie'
ORDER BY embedding <=> $1
LIMIT 5;

-- name: binary_rescore
-- Binaire + re-scoring : 1) 40 candidats par distance de Hamming sur l'index binaire (rapide, compact),
-- 2) re-classement exact en float32 sur ces seuls candidats.
SELECT id, article_id, 1 - (embedding <=> $1) AS similarity
FROM (
    SELECT * FROM chunks
    ORDER BY binary_quantize(embedding)::bit(384) <~> binary_quantize($1::vector)  -- cast obligatoire
    LIMIT 40
) AS candidates
ORDER BY embedding <=> $1
LIMIT 5;

-- name: hybrid_rrf
-- Hybride en SQL : classements dense et plein texte fusionnés par RRF (k = 60), en une requête.
-- $2 = la question en texte.
WITH dense AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> $1) AS rank
    FROM chunks ORDER BY embedding <=> $1 LIMIT 20
), lexical AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(content_tsv, query) DESC) AS rank
    FROM chunks, websearch_to_tsquery('french', $2) AS query
    WHERE content_tsv @@ query
    ORDER BY ts_rank_cd(content_tsv, query) DESC LIMIT 20
)
SELECT c.id, c.article_id,
       coalesce(1.0 / (60 + d.rank), 0) + coalesce(1.0 / (60 + l.rank), 0) AS rrf
FROM chunks c
LEFT JOIN dense d ON d.id = c.id
LEFT JOIN lexical l ON l.id = c.id
WHERE d.id IS NOT NULL OR l.id IS NOT NULL
ORDER BY rrf DESC
LIMIT 5;
