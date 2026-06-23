-- hermes-cmx Postgres schema — applied automatically on first container boot.
-- Mirrors the SQLite VerbatimStore (verbatim messages + FTS + trigram + embeddings +
-- session lineage), but with real concurrent writes (MVCC) and proper indexes.

CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram similarity (substring/identifier recall)
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector (embeddings)

-- ── verbatim messages ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    session_id    TEXT   NOT NULL,
    turn_index    INTEGER NOT NULL,
    role          TEXT   NOT NULL,
    content       TEXT   NOT NULL,
    content_hash  TEXT   NOT NULL,
    token_count   INTEGER,
    pinned        BOOLEAN NOT NULL DEFAULT FALSE,
    externalized_ref TEXT,
    created_at    DOUBLE PRECISION NOT NULL,
    -- generated FTS vector (full-text); STORED so it indexes + stays in sync on write
    fts           tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);

CREATE INDEX IF NOT EXISTS idx_messages_session  ON messages (session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_messages_pinned   ON messages (session_id) WHERE pinned;
-- FTS: GIN over the generated tsvector
CREATE INDEX IF NOT EXISTS idx_messages_fts      ON messages USING gin (fts);
-- trigram: GIN over content for substring/fuzzy recall (the pg_trgm analogue of the
-- SQLite trigram tokenizer)
CREATE INDEX IF NOT EXISTS idx_messages_trgm     ON messages USING gin (content gin_trgm_ops);
-- dedup helper (content-hash per session); not UNIQUE on purpose (short identical turns
-- may legitimately repeat — matches the SQLite store's behavior)
CREATE INDEX IF NOT EXISTS idx_messages_hash     ON messages (session_id, content_hash);

-- ── embeddings (pgvector). Dimension is set at first use via ALTER if needed; default
-- 1536 for text-embedding-3-small. A separate table keeps message rows lean and lets the
-- embedding be written asynchronously after ingest. ──────────────────────────────────
CREATE TABLE IF NOT EXISTS message_embeddings (
    message_id  BIGINT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    session_id  TEXT NOT NULL,
    embedding   vector(1536)
);
CREATE INDEX IF NOT EXISTS idx_emb_session ON message_embeddings (session_id);
-- HNSW for fast cosine ANN (created here; pgvector builds it incrementally)
CREATE INDEX IF NOT EXISTS idx_emb_hnsw ON message_embeddings
    USING hnsw (embedding vector_cosine_ops);

-- ── session lineage (Hermes rotates session_id on compaction; map child -> root) ──────
CREATE TABLE IF NOT EXISTS session_lineage (
    session_id TEXT PRIMARY KEY,
    root_id    TEXT NOT NULL
);
