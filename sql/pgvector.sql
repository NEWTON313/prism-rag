-- Prism — schéma pgvector de référence (leçons 2.2 à 2.4 et 3.1). Testé avec PGlite + pgvector.
-- Dimension 384 : celle du modèle d'embeddings du cours (paraphrase-multilingual-MiniLM-L12-v2).

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Les passages, leur client (tenant), leurs métadonnées et leur embedding ------------------
CREATE TABLE chunks (
    id          text PRIMARY KEY,
    tenant_id   text NOT NULL,                 -- multi-tenant : chaque client ne voit que ses données
    article_id  text NOT NULL,
    category    text NOT NULL,
    content     text NOT NULL,
    embedding   vector(384) NOT NULL,
    -- recherche lexicale
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('french', content)) STORED
);

-- 2. Index HNSW sur la distance cosinus --------------------------------------------------------
-- m : voisins par nœud (16 par défaut) ; ef_construction : effort de construction (64 par défaut)
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tenant ON chunks (tenant_id, category);   -- pour les filtres
CREATE INDEX chunks_tsv ON chunks USING gin (content_tsv);

-- 3. Quantization : demi-précision (2 octets/dim) et binaire (1 bit/dim), en index d'expression --
CREATE INDEX chunks_embedding_half ON chunks
    USING hnsw ((embedding::halfvec(384)) halfvec_cosine_ops);
CREATE INDEX chunks_embedding_bin ON chunks
    USING hnsw ((binary_quantize(embedding)::bit(384)) bit_hamming_ops);

-- 4. Isolation des clients par Row Level Security ---------------------------------------------
-- L'application fixe le client de la requête : SET app.tenant_id = 'nimbus'; puis toutes les requêtes
-- sont filtrées par la base elle-même, même si le code oublie le WHERE.
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = current_setting('app.tenant_id', true));
