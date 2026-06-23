# 01 — Architecture

## 1. Component map

```
                          ┌─────────────────────────────────────────────────────────┐
                          │                     hermes-cmx engine                     │
                          │  (implements Hermes ContextEngine ABC; one active engine) │
                          └─────────────────────────────────────────────────────────┘
  Ingest path                         Assembly path (per turn)              Enforcement path (per answer)
  ───────────                         ──────────────────────────            ────────────────────────────
  user/assistant/tool turn            1. Pinned sinks                       A. Parse citations
        │                             2. Recent verbatim window             B. Deterministic citation check (SQLite)
        ▼                             3. RETRIEVED EVIDENCE (hybrid)         C. Independent verification pass
  ┌──────────────┐                    4. (optional) headnote pointers       D. Forced-tool / regenerate loop
  │ Verbatim WAL │◄──────┐            5. budget-fit to model window         E. refuse-to-guess fallback
  │  (SQLite)    │       │                     │                                    │
  └──────┬───────┘       │                     ▼                                    ▼
         │ index         │            ┌──────────────────┐                  grounded answer shipped
         ▼               │            │  Retrieval core  │                  + citation audit persisted
  ┌──────────────────┐   │            │ FTS5 + trigram + │
  │ Indexers         │   │            │ embeddings (fuse)│◄─── model tools: cmx_recall / cmx_expand /
  │ • messages_fts   │   │            └──────────────────┘     cmx_grep / cmx_graph_query
  │ • messages_trgm  │   │                     ▲
  │ • embeddings     │───┘                     │
  │ • entities/graph │            provider-aware tokenizer + enforcement profile (per model)
  │ • externalizer   │
  └──────────────────┘
```

Two facts hold everything together:
- **Only one context engine is active** in Hermes (config `context.engine`). cmx replaces LCM cleanly.
- **The verbatim WAL is the single source of truth.** Every other table is a derived, rebuildable index.

## 2. Storage: SQLite schema

All tables in `<HERMES_HOME>/cmx.db`. The verbatim store is **append-only and immutable**; compaction never deletes rows (only optional externalization of huge payloads, content preserved on disk).

### 2.1 Verbatim store (truth)
```sql
CREATE TABLE messages (
  id            INTEGER PRIMARY KEY,          -- stable citation handle
  session_id    TEXT NOT NULL,
  turn_index    INTEGER NOT NULL,
  role          TEXT NOT NULL,                -- user|assistant|tool|system
  content       TEXT NOT NULL,                -- verbatim; never paraphrased
  content_hash  TEXT NOT NULL,                -- dedupe / integrity
  token_count   INTEGER,                      -- per provider-aware tokenizer at ingest
  pinned        INTEGER DEFAULT 0,            -- sink candidate (system/identity/task)
  externalized_ref TEXT,                      -- path if payload offloaded; content recoverable
  created_at    TEXT NOT NULL
);
CREATE INDEX idx_messages_session ON messages(session_id, turn_index);
```

### 2.2 Retrieval accelerators (rebuildable)
```sql
-- (a) word-level full text (BM25)
CREATE VIRTUAL TABLE messages_fts USING fts5(
  content, content='messages', content_rowid='id', tokenize='unicode61 remove_diacritics 2');

-- (b) trigram: substring / identifier / typo-tolerant (the LCM gap)
CREATE VIRTUAL TABLE messages_trgm USING fts5(
  content, content='messages', content_rowid='id', tokenize='trigram');

-- (c) chunking for long messages so retrieval granularity < whole message
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, message_id INTEGER NOT NULL,
  span_start INTEGER, span_end INTEGER, text TEXT NOT NULL, token_count INTEGER);

-- (d) dense vectors (sqlite-vec / sqlite-vss / faiss sidecar). Optional + degradable.
CREATE TABLE embeddings (
  chunk_id INTEGER PRIMARY KEY, vector BLOB NOT NULL, dim INTEGER, model TEXT);
```
> **Trigram note (platform):** SQLite's FTS5 `trigram` tokenizer requires a recent SQLite (≥ 3.34) built with FTS5. On the user's OL8 / aarch64 host this needs the bundled `pysqlite3`/custom SQLite (see `HERMES_SQLITE_DRIVER=pysqlite3` already in their env, and the `sqlite-trigram-ol8-aarch64` runbook). Phase 1 must verify the trigram tokenizer loads on the live host before depending on it; FTS5-word + LIKE is the documented fallback.

### 2.3 Structural index (multi-hop, optional layer)
```sql
CREATE TABLE entities  (id INTEGER PRIMARY KEY, kind TEXT, name TEXT, norm TEXT);
CREATE TABLE relations (id INTEGER PRIMARY KEY, src INTEGER, rel TEXT, dst INTEGER, message_id INTEGER);
CREATE TABLE mentions  (entity_id INTEGER, message_id INTEGER);   -- entity ↔ verbatim row
-- "headnotes": SHORT navigational pointers, NEVER prompt-substitutes for verbatim
CREATE TABLE headnotes (id INTEGER PRIMARY KEY, session_id TEXT,
  turn_lo INTEGER, turn_hi INTEGER, label TEXT, message_ids TEXT);  -- e.g. "turns 1200-1250: dashboard CI4 migration"
```

### 2.4 Grounding audit (the proof)
```sql
CREATE TABLE answer_claims (
  id INTEGER PRIMARY KEY, session_id TEXT, turn_index INTEGER,
  claim_span TEXT,                 -- the factual assertion as written
  cited_ids TEXT,                  -- message ids the model cited
  verified INTEGER,                -- 1 verified / 0 failed / NULL n.a.
  method TEXT,                     -- citation_exact | verifier_llm | refused
  created_at TEXT);
CREATE TABLE pins (session_id TEXT, message_id INTEGER, reason TEXT);  -- sinks
```

Every shipped factual claim leaves a row here → fully auditable grounding.

## 3. Retrieval core (the floor of the whole design)

Retrieval precision is the single most important quality lever: if the right slice is not surfaced, no downstream enforcement can save the answer. cmx uses **hybrid fusion**:

```
query (entities/identifiers extracted from the user turn + recent context)
   ├─ FTS5/BM25       → word-level relevance
   ├─ trigram         → substrings, identifiers (CI4_migrate), typos, code/log tokens
   └─ embeddings(kNN) → semantic / paraphrased matches
            │
        Reciprocal-Rank Fusion (RRF) + recency prior + pin boost
            │
        top-K verbatim slices (message_id + span), de-duplicated, budget-trimmed
```

- **RRF** avoids tuning brittle score weights across three very different scorers.
- **Recency prior** and **pin boost** are additive, bounded nudges — never override a strong content match.
- Output always carries `message_id` (+ chunk span) so injected evidence is **citable** and the deterministic check (layer 3) has a target.
- Degrades gracefully: embeddings optional (FTS5+trigram only); trigram optional (FTS5+LIKE only — LCM's floor).

## 4. The turn lifecycle (where enforcement lives)

This is the spine. Steps 2 and 5 are what make cmx different from LCM.

```
INGEST
  1. Append user turn to messages (verbatim). Index: fts, trigram, chunk, embed (async ok), extract entities.

ASSEMBLE  (engine, budget-aware to the model's REAL window)
  2a. Pinned sinks            : system prompt + IDENTITY + active task spec (never evicted)
  2b. Recent verbatim window  : last N turns, full (N scaled to window)
  2c. RETRIEVED EVIDENCE      : hybrid retrieval keyed on this turn → top-K verbatim slices,
                                labelled "[CMX EVIDENCE — verbatim from your history; cite ids]"
                                (engine/tool role — NEVER assistant role)
  2d. (optional) headnotes    : short pointers to regions the model may expand
  2e. budget-fit              : trim 2c→2b→2d in priority order to fit the provider tokenizer budget

SEND  (with the model's enforcement profile)
  3. For factual-answer turns, apply the profile: tool_choice gate / required citation format / verifier on|off

MODEL LOOP
  4. Model answers, or calls cmx_recall/cmx_expand/cmx_grep/cmx_graph_query → engine serves verbatim → loop

ENFORCE  (engine, per answer — no model goodwill required)
  5a. Parse citations from the answer
  5b. Deterministic citation check: each cited id exists AND the quoted span occurs in that verbatim row
  5c. Independent verification pass: a separate (often cheaper) model/heuristic checks factual claims
      that lack support in 2c/citations
  5d. On failure → forced-retrieval regenerate (bounded retries) → else refuse-to-guess downgrade
  6. Ship grounded answer; write answer_claims audit rows.
```

## 5. Model-facing tools (constrained, engine-served)

| Tool | Purpose | Returns |
|---|---|---|
| `cmx_recall(range\|session)` | pull a contiguous verbatim span | verbatim rows + ids |
| `cmx_expand(id[, span])` | rehydrate a specific message/chunk | verbatim text + id |
| `cmx_grep(query)` | hybrid search (FTS5+trigram+embeddings) | ranked slices + ids |
| `cmx_graph_query(entity)` | multi-hop over entity/relation graph | linked ids + slices |

All return **verbatim with ids**, so anything the model uses is immediately citable and checkable. There is deliberately **no** `cmx_summarize` that produces prompt-substitutes.

## 6. Configuration

cmx reads, in precedence order: `CMX_*` env vars → a real `cmx:` block in `~/.hermes/config.yaml` (the LCM gap we fix) → defaults. Provider-aware tokenizer and enforcement profiles are config-driven and overridable per model. See `05-IMPLEMENTATION-PLAN.md` §Host-integration.
