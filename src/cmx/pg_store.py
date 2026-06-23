"""Postgres backend for the verbatim store — same interface as cmx.store.VerbatimStore,
but on Postgres for true concurrent reads AND writes (multi-tab / multi-subagent / council
fan-out). FTS via tsvector, substring/identifier recall via pg_trgm word_similarity,
embeddings as bytea (dim-agnostic; reuses the numpy pack/unpack + cosine path unchanged).

Drop-in: HybridRetriever / assembly / hermes_engine consume only the public methods below,
so nothing downstream changes when this backend is selected.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional

from . import substrate

_TOKEN = re.compile(r"[0-9A-Za-z_]+", re.UNICODE)

_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS messages (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  session_id    TEXT NOT NULL,
  turn_index    INTEGER NOT NULL,
  role          TEXT NOT NULL,
  content       TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  token_count   INTEGER,
  pinned        BOOLEAN NOT NULL DEFAULT FALSE,
  externalized_ref TEXT,
  created_at    DOUBLE PRECISION NOT NULL,
  fts           tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_messages_fts  ON messages USING gin(fts);
CREATE INDEX IF NOT EXISTS idx_messages_trgm ON messages USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_messages_hash ON messages(session_id, content_hash);

CREATE TABLE IF NOT EXISTS embeddings (
  message_id BIGINT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
  vector BYTEA NOT NULL, dim INTEGER, model TEXT);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id   BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  parent_id  BIGINT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  session_id TEXT NOT NULL,
  vector     BYTEA NOT NULL, dim INTEGER, model TEXT);
CREATE INDEX IF NOT EXISTS idx_chunk_emb_session ON chunk_embeddings(session_id);
CREATE INDEX IF NOT EXISTS idx_chunk_emb_parent  ON chunk_embeddings(parent_id);

CREATE TABLE IF NOT EXISTS session_lineage (
  session_id TEXT PRIMARY KEY,
  root_id    TEXT NOT NULL);

-- Fan-out lineage (30-worker council): record child chats under a ROOT deliberation for
-- ORGANIZATION/TRACKING only. Unlike session_lineage (which merges retrieval across
-- compaction rollovers), this does NOT merge retrieval — workers stay isolated/unbiased.
-- The synthesizer/root uses it to enumerate the tree; each worker still retrieves only its
-- own session + the shared :findings namespace.
CREATE TABLE IF NOT EXISTS fanout_lineage (
  child_id   TEXT PRIMARY KEY,
  root_id    TEXT NOT NULL,
  parent_id  TEXT,
  role       TEXT NOT NULL DEFAULT '',
  created_at DOUBLE PRECISION NOT NULL);
CREATE INDEX IF NOT EXISTS idx_fanout_root ON fanout_lineage(root_id);
"""

_COLS = ("id", "session_id", "turn_index", "role", "content", "content_hash",
         "token_count", "pinned", "externalized_ref", "created_at")
_SELECT = "SELECT " + ", ".join(_COLS) + " FROM messages"


def _terms(text: str, min_len: int) -> list[str]:
    seen, out = set(), []
    for t in _TOKEN.findall((text or "").lower()):
        if len(t) < min_len or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 32:
            break
    return out


class PostgresVerbatimStore:
    """Same public surface as VerbatimStore, backed by Postgres."""

    def __init__(self, dsn: str, *, use_trigram: bool = True, use_graph: bool = True,
                 min_size: int = 1, max_size: int = 10):
        self.dsn = dsn
        self.pool = substrate.get_pool(dsn, min_size=min_size, max_size=max_size)
        with self.pool.connection() as conn:
            conn.execute(_SCHEMA)
            conn.commit()
        # pg_trgm is always available here (extension created above).
        self.has_trigram = bool(use_trigram)
        # Graph multi-hop stays SQLite-only for now; None cleanly disables it downstream.
        self.graph = None

    # -- write (append-only) ---------------------------------------------
    def add_message(self, session_id: str, turn_index: int, role: str, content: str,
                    *, pinned: bool = False, token_count: Optional[int] = None,
                    externalized_ref: Optional[str] = None) -> int:
        h = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        with self.pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO messages(session_id,turn_index,role,content,content_hash,"
                "token_count,pinned,externalized_ref,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (session_id, turn_index, role, content, h, token_count,
                 bool(pinned), externalized_ref, time.time()),
            ).fetchone()
            conn.commit()
            return int(row["id"])

    def set_pinned(self, message_id: int, pinned: bool = True) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE messages SET pinned=%s WHERE id=%s", (bool(pinned), message_id))
            conn.commit()

    # -- read -------------------------------------------------------------
    def get_message(self, message_id: int) -> Optional[dict]:
        with self.pool.connection() as conn:
            r = conn.execute(_SELECT + " WHERE id=%s", (message_id,)).fetchone()
        return dict(r) if r else None

    def recent(self, session_id: str, n: int) -> list[dict]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                _SELECT + " WHERE session_id=%s ORDER BY turn_index DESC, id DESC LIMIT %s",
                (session_id, n)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def pinned(self, session_id: str) -> list[dict]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                _SELECT + " WHERE session_id=%s AND pinned=TRUE ORDER BY turn_index",
                (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def count(self, session_id: Optional[str] = None) -> int:
        with self.pool.connection() as conn:
            if session_id is None:
                r = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()
            else:
                r = conn.execute("SELECT COUNT(*) c FROM messages WHERE session_id=%s",
                                 (session_id,)).fetchone()
        return int(r["c"])

    def max_turn(self, session_id: str) -> int:
        with self.pool.connection() as conn:
            r = conn.execute(
                "SELECT COALESCE(MAX(turn_index),-1) m FROM messages WHERE session_id=%s",
                (session_id,)).fetchone()
        return int(r["m"])

    def in_session(self, message_id: int, session_id: str) -> bool:
        with self.pool.connection() as conn:
            r = conn.execute("SELECT 1 FROM messages WHERE id=%s AND session_id=%s",
                             (message_id, session_id)).fetchone()
        return r is not None

    def content_hashes(self, session_id: str) -> set:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT content_hash FROM messages WHERE session_id=%s", (session_id,)).fetchall()
        return {r["content_hash"] for r in rows}

    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8", "replace")).hexdigest()

    # -- session lineage --------------------------------------------------
    def root_session(self, session_id: str) -> str:
        if not session_id:
            return session_id
        with self.pool.connection() as conn:
            r = conn.execute("SELECT root_id FROM session_lineage WHERE session_id=%s",
                             (session_id,)).fetchone()
        return r["root_id"] if r else session_id

    def link_session(self, child_id: str, old_id: str) -> str:
        if not child_id:
            return self.root_session(old_id)
        root = self.root_session(old_id) if old_id else child_id
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO session_lineage(session_id, root_id) VALUES(%s,%s) "
                "ON CONFLICT (session_id) DO UPDATE SET root_id=EXCLUDED.root_id",
                (child_id, root))
            if old_id and old_id != root:
                conn.execute(
                    "INSERT INTO session_lineage(session_id, root_id) VALUES(%s,%s) "
                    "ON CONFLICT (session_id) DO NOTHING", (old_id, root))
            conn.commit()
        return root

    # -- embeddings (message-level) --------------------------------------
    def store_embeddings(self, pairs, *, model: str = "", dim: int = 0) -> None:
        from .embeddings import pack
        rows = [(int(mid), pack(v), len(v), model) for mid, v in pairs]
        if not rows:
            return
        with self.pool.connection() as conn:
            conn.cursor().executemany(
                "INSERT INTO embeddings(message_id,vector,dim,model) VALUES(%s,%s,%s,%s) "
                "ON CONFLICT (message_id) DO UPDATE SET vector=EXCLUDED.vector, "
                "dim=EXCLUDED.dim, model=EXCLUDED.model", rows)
            conn.commit()

    def messages_missing_embeddings(self, session_id: str) -> list[tuple[int, str]]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT m.id AS id, m.content AS content FROM messages m "
                "LEFT JOIN embeddings e ON e.message_id = m.id "
                "WHERE m.session_id = %s AND e.message_id IS NULL", (session_id,)).fetchall()
        return [(int(r["id"]), r["content"]) for r in rows]

    def get_session_vectors(self, session_id: str):
        import numpy as np
        from .embeddings import unpack
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT e.message_id AS id, e.vector AS v FROM embeddings e "
                "JOIN messages m ON m.id = e.message_id WHERE m.session_id = %s "
                "ORDER BY e.message_id", (session_id,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        ids = [int(r["id"]) for r in rows]
        mat = np.vstack([unpack(bytes(r["v"])) for r in rows])
        return ids, mat

    # -- chunk-level embeddings ------------------------------------------
    def messages_missing_chunk_embeddings(self, session_id: str, min_chars: int):
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT m.id AS id, m.content AS content FROM messages m "
                "WHERE m.session_id = %s AND length(m.content) >= %s "
                "AND NOT EXISTS (SELECT 1 FROM chunk_embeddings c WHERE c.parent_id = m.id)",
                (session_id, int(min_chars))).fetchall()
        return [(int(r["id"]), r["content"]) for r in rows]

    def store_chunk_embeddings(self, session_id: str, triples, *, model: str = "", dim: int = 0):
        from .embeddings import pack
        rows = [(int(pid), session_id, pack(v), len(v), model) for pid, v in triples]
        if not rows:
            return
        with self.pool.connection() as conn:
            conn.cursor().executemany(
                "INSERT INTO chunk_embeddings(parent_id,session_id,vector,dim,model) "
                "VALUES(%s,%s,%s,%s,%s)", rows)
            conn.commit()

    def get_session_chunk_vectors(self, session_id: str):
        import numpy as np
        from .embeddings import unpack
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT parent_id AS pid, vector AS v FROM chunk_embeddings "
                "WHERE session_id = %s ORDER BY chunk_id", (session_id,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        pids = [int(r["pid"]) for r in rows]
        mat = np.vstack([unpack(bytes(r["v"])) for r in rows])
        return pids, mat

    # -- search -----------------------------------------------------------
    def search_fts(self, query: str, k: int = 20, session_id: Optional[str] = None) -> list[dict]:
        terms = _terms(query, 2)
        if not terms:
            return []
        tsq = " | ".join(terms)
        clause = "" if session_id is None else " AND m.session_id = %(sid)s"
        sql = (
            "SELECT m.id AS id, ts_rank(m.fts, to_tsquery('simple', %(q)s)) AS score "
            "FROM messages m WHERE m.fts @@ to_tsquery('simple', %(q)s)" + clause +
            " ORDER BY score DESC LIMIT %(k)s")
        try:
            with self.pool.connection() as conn:
                rows = conn.execute(sql, {"q": tsq, "sid": session_id, "k": k}).fetchall()
        except Exception:
            return []
        return [{"id": int(r["id"]), "score": float(r["score"])} for r in rows]

    def search_trgm(self, query: str, k: int = 20, session_id: Optional[str] = None) -> list[dict]:
        if not self.has_trigram:
            return []
        terms = _terms(query, 3)
        if not terms:
            return []
        needle = " ".join(terms)
        clause = "" if session_id is None else " AND m.session_id = %(sid)s"
        # word_similarity (<%) finds the best-matching substring/word — the correct analogue
        # of the SQLite trigram tokenizer (plain % whole-string similarity is too strict).
        sql = (
            "SELECT m.id AS id, word_similarity(%(q)s, m.content) AS score "
            "FROM messages m WHERE %(q)s <%% m.content" + clause +
            " ORDER BY score DESC LIMIT %(k)s")
        try:
            with self.pool.connection() as conn:
                rows = conn.execute(sql, {"q": needle, "sid": session_id, "k": k}).fetchall()
        except Exception:
            return []
        return [{"id": int(r["id"]), "score": float(r["score"])} for r in rows]

    def close(self) -> None:
        # The pool is shared across stores/domains for this DSN — do not tear it down here.
        # Use substrate.close_all() at process shutdown.
        pass

    # -- fan-out lineage (30-worker council tracking; NOT retrieval-merged) ----
    def register_fanout(self, root_id: str, child_id: str, *, parent_id: Optional[str] = None,
                        role: str = "") -> None:
        """Record that ``child_id`` is a worker/sub-chat of the ``root_id`` deliberation.
        Tracking only — does not merge retrieval, so the worker stays isolated/unbiased."""
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO fanout_lineage(child_id,root_id,parent_id,role,created_at) "
                "VALUES(%s,%s,%s,%s,%s) ON CONFLICT (child_id) DO UPDATE SET "
                "root_id=EXCLUDED.root_id, parent_id=EXCLUDED.parent_id, role=EXCLUDED.role",
                (child_id, root_id, parent_id, role, time.time()))
            conn.commit()

    def fanout_children(self, root_id: str) -> list[dict]:
        """All children (workers/personas) recorded under a root deliberation."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT child_id, parent_id, role, created_at FROM fanout_lineage "
                "WHERE root_id=%s ORDER BY created_at", (root_id,)).fetchall()
        return [dict(r) for r in rows]

    def fanout_root(self, child_id: str) -> Optional[str]:
        """The root deliberation a child belongs to (None if not a fan-out child)."""
        with self.pool.connection() as conn:
            r = conn.execute("SELECT root_id FROM fanout_lineage WHERE child_id=%s",
                             (child_id,)).fetchone()
        return r["root_id"] if r else None
