"""Verbatim store — the single source of truth.

Append-only, immutable ``messages`` table mirrored into two FTS5 indexes
(word-level unicode61 + substring/identifier-friendly trigram). Compaction never
deletes message rows; only explicit session purge does (delete trigger keeps FTS
in sync). Everything else is a rebuildable accelerator.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional

from . import db

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id            INTEGER PRIMARY KEY,
  session_id    TEXT NOT NULL,
  turn_index    INTEGER NOT NULL,
  role          TEXT NOT NULL,
  content       TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  token_count   INTEGER,
  pinned        INTEGER DEFAULT 0,
  externalized_ref TEXT,
  created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, turn_index);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content, content='messages', content_rowid='id',
  tokenize='unicode61');

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TABLE IF NOT EXISTS embeddings (
  message_id INTEGER PRIMARY KEY, vector BLOB NOT NULL, dim INTEGER, model TEXT);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
  chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
  parent_id  INTEGER NOT NULL,
  session_id TEXT NOT NULL,
  vector     BLOB NOT NULL, dim INTEGER, model TEXT);
CREATE INDEX IF NOT EXISTS idx_chunk_emb_session ON chunk_embeddings(session_id);
CREATE INDEX IF NOT EXISTS idx_chunk_emb_parent ON chunk_embeddings(parent_id);

-- Session lineage: Hermes rotates session_id on compaction/rollover (passing the old
-- id as parent). Without this, a session-scoped store would lose the whole prior
-- conversation the moment it compacts. We map every child id → its lineage ROOT and
-- store/retrieve under the root, so one logical conversation stays one session_id
-- across any number of rollovers. Survives process restarts (resumed child resolves
-- to its root). Append-only friendly: no message rows are ever rewritten.
CREATE TABLE IF NOT EXISTS session_lineage (
  session_id TEXT PRIMARY KEY,
  root_id    TEXT NOT NULL);
"""

_SCHEMA_TRIGRAM = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_trgm USING fts5(
  content, content='messages', content_rowid='id', tokenize='trigram');
CREATE TRIGGER IF NOT EXISTS messages_trgm_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_trgm(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_trgm_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_trgm(messages_trgm, rowid, content) VALUES('delete', old.id, old.content);
END;
"""

_TOKEN = re.compile(r"[0-9A-Za-z_./:-]{2,}", re.UNICODE)


def _row(r) -> dict:
    return {k: r[k] for k in r.keys()}


def _fts_query(text: str, *, min_len: int = 2) -> Optional[str]:
    """Build a safe FTS5 OR-of-quoted-terms query. Returns None if no usable term."""
    terms = [t for t in _TOKEN.findall(text or "") if len(t) >= min_len]
    if not terms:
        return None
    seen, out = set(), []
    for t in terms[:32]:
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append('"' + t.replace('"', '""') + '"')
    return " OR ".join(out) if out else None


class VerbatimStore:
    def __init__(self, path: str, *, use_trigram: bool = True, use_graph: bool = True):
        self.conn = db.connect(path)
        self.conn.executescript(_SCHEMA)
        self.has_trigram = False
        if use_trigram:
            try:
                self.conn.executescript(_SCHEMA_TRIGRAM)
                self.has_trigram = True
            except Exception:
                self.has_trigram = False
        self.graph = None
        if use_graph:
            try:
                from .graph import GraphIndex
                self.graph = GraphIndex(self.conn)
            except Exception:
                self.graph = None

    # -- write (append-only) ---------------------------------------------
    def add_message(self, session_id: str, turn_index: int, role: str, content: str,
                    *, pinned: bool = False, token_count: Optional[int] = None,
                    externalized_ref: Optional[str] = None) -> int:
        h = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()
        cur = self.conn.execute(
            "INSERT INTO messages(session_id,turn_index,role,content,content_hash,"
            "token_count,pinned,externalized_ref,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (session_id, turn_index, role, content, h, token_count,
             1 if pinned else 0, externalized_ref, time.time()),
        )
        mid = int(cur.lastrowid)
        if self.graph is not None:
            try:
                self.graph.index(mid, content)
            except Exception:
                pass
        return mid

    def set_pinned(self, message_id: int, pinned: bool = True) -> None:
        self.conn.execute("UPDATE messages SET pinned=? WHERE id=?",
                          (1 if pinned else 0, message_id))

    # -- read -------------------------------------------------------------
    def get_message(self, message_id: int) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
        return _row(r) if r else None

    def recent(self, session_id: str, n: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY turn_index DESC, id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        return [_row(r) for r in reversed(rows)]

    def pinned(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id=? AND pinned=1 ORDER BY turn_index", (session_id,)
        ).fetchall()
        return [_row(r) for r in rows]

    def count(self, session_id: Optional[str] = None) -> int:
        if session_id is None:
            return int(self.conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"])
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE session_id=?", (session_id,)).fetchone()["c"])

    def max_turn(self, session_id: str) -> int:
        r = self.conn.execute(
            "SELECT COALESCE(MAX(turn_index),-1) m FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()
        return int(r["m"])

    def in_session(self, message_id: int, session_id: str) -> bool:
        r = self.conn.execute("SELECT 1 FROM messages WHERE id=? AND session_id=?",
                              (message_id, session_id)).fetchone()
        return r is not None

    def content_hashes(self, session_id: str) -> set:
        """Set of content_hash already stored for a session — lets the incremental
        ingest be idempotent (dedup) instead of a fragile positional cursor that breaks
        when the host replaces the message list across a compaction."""
        rows = self.conn.execute(
            "SELECT content_hash FROM messages WHERE session_id=?", (session_id,)).fetchall()
        return {r["content_hash"] for r in rows}

    @staticmethod
    def hash_content(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8", "replace")).hexdigest()

    # -- session lineage (compaction/rollover continuity) -----------------
    def root_session(self, session_id: str) -> str:
        """Resolve a session_id to its lineage ROOT (the first id of the logical
        conversation). A standalone session is its own root."""
        if not session_id:
            return session_id
        r = self.conn.execute(
            "SELECT root_id FROM session_lineage WHERE session_id=?", (session_id,)).fetchone()
        return r["root_id"] if r else session_id

    def link_session(self, child_id: str, old_id: str) -> str:
        """Record that ``child_id`` continues the conversation rooted at ``old_id``'s root
        (Hermes compaction rollover). Returns the resolved root id. Idempotent."""
        if not child_id:
            return self.root_session(old_id)
        root = self.root_session(old_id) if old_id else child_id
        self.conn.execute(
            "INSERT OR REPLACE INTO session_lineage(session_id, root_id) VALUES(?,?)",
            (child_id, root))
        if old_id and old_id != root:
            self.conn.execute(
                "INSERT OR IGNORE INTO session_lineage(session_id, root_id) VALUES(?,?)",
                (old_id, root))
        return root

    # -- embeddings (iteration-2 lever 1) --------------------------------
    def store_embeddings(self, pairs, *, model: str = "", dim: int = 0) -> None:
        from .embeddings import pack
        rows = [(int(mid), pack(v), len(v), model) for mid, v in pairs]
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings(message_id,vector,dim,model) VALUES(?,?,?,?)",
                rows)

    def messages_missing_embeddings(self, session_id: str) -> list[tuple[int, str]]:
        rows = self.conn.execute(
            "SELECT m.id AS id, m.content AS content FROM messages m "
            "LEFT JOIN embeddings e ON e.message_id = m.id "
            "WHERE m.session_id = ? AND e.message_id IS NULL", (session_id,)).fetchall()
        return [(int(r["id"]), r["content"]) for r in rows]

    def get_session_vectors(self, session_id: str):
        import numpy as np
        from .embeddings import unpack
        rows = self.conn.execute(
            "SELECT e.message_id AS id, e.vector AS v FROM embeddings e "
            "JOIN messages m ON m.id = e.message_id WHERE m.session_id = ? "
            "ORDER BY e.message_id", (session_id,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        ids = [int(r["id"]) for r in rows]
        mat = np.vstack([unpack(r["v"]) for r in rows])
        return ids, mat

    # -- chunk-level embeddings (iter3 L3) --------------------------------
    def messages_missing_chunk_embeddings(self, session_id: str, min_chars: int):
        rows = self.conn.execute(
            "SELECT m.id AS id, m.content AS content FROM messages m "
            "WHERE m.session_id = ? AND length(m.content) >= ? "
            "AND NOT EXISTS (SELECT 1 FROM chunk_embeddings c WHERE c.parent_id = m.id)",
            (session_id, int(min_chars))).fetchall()
        return [(int(r["id"]), r["content"]) for r in rows]

    def store_chunk_embeddings(self, session_id: str, triples, *, model: str = "", dim: int = 0):
        from .embeddings import pack
        rows = [(int(pid), session_id, pack(v), len(v), model) for pid, v in triples]
        if rows:
            self.conn.executemany(
                "INSERT INTO chunk_embeddings(parent_id,session_id,vector,dim,model) "
                "VALUES(?,?,?,?,?)", rows)

    def get_session_chunk_vectors(self, session_id: str):
        import numpy as np
        from .embeddings import unpack
        rows = self.conn.execute(
            "SELECT parent_id AS pid, vector AS v FROM chunk_embeddings "
            "WHERE session_id = ? ORDER BY chunk_id", (session_id,)).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype="float32")
        pids = [int(r["pid"]) for r in rows]
        mat = np.vstack([unpack(r["v"]) for r in rows])
        return pids, mat

    # -- search -----------------------------------------------------------
    def _search(self, table: str, query: str, k: int, session_id: Optional[str],
                min_len: int) -> list[dict]:
        q = _fts_query(query, min_len=min_len)
        if not q:
            return []
        clause = "" if session_id is None else " AND m.session_id = :sid"
        sql = (
            f"SELECT m.id AS id, bm25({table}) AS score "
            f"FROM {table} JOIN messages m ON m.id = {table}.rowid "
            f"WHERE {table} MATCH :q{clause} ORDER BY score ASC LIMIT :k"
        )
        try:
            rows = self.conn.execute(sql, {"q": q, "sid": session_id, "k": k}).fetchall()
        except Exception:
            return []
        return [{"id": int(r["id"]), "score": float(r["score"])} for r in rows]

    def search_fts(self, query: str, k: int = 20, session_id: Optional[str] = None) -> list[dict]:
        return self._search("messages_fts", query, k, session_id, min_len=2)

    def search_trgm(self, query: str, k: int = 20, session_id: Optional[str] = None) -> list[dict]:
        if not self.has_trigram:
            return []
        return self._search("messages_trgm", query, k, session_id, min_len=3)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
