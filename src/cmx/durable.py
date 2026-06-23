"""Stage 1 — durable cross-session memory ingest (verbatim, no-drop).

Durable memory (e.g. ``MEMORY.md`` / ``USER.md``) is stored verbatim in the cmx store under a
FIXED namespace (``cfg.durable_session``) that never rotates with the conversation. cmx then
retrieves the turn-relevant slices each turn (``assembly.assemble``) and exposes the full content
on demand via ``cmx_grep`` — so nothing is ever dropped or summarized; the budget only bounds how
much is *proactively* injected.

Chunking keeps retrieval granular: a 600KB file becomes many small, individually-retrievable
verbatim rows. Re-ingest is idempotent (content-hash dedup), so loading on every engine
construction is cheap and safe.
"""
from __future__ import annotations

import os
from typing import Any, List

from .store import VerbatimStore


def chunk_document(text: str, target_chars: int = 700) -> List[str]:
    """Split a document into retrieval-sized verbatim chunks.

    Splits on blank lines (paragraph/section boundaries) and accumulates blocks up to
    ``target_chars``; a single oversized block is split on line boundaries so no chunk is
    pathologically large. Verbatim — text is never altered, only segmented.
    """
    if not text or not text.strip():
        return []
    blocks = [b for b in text.replace("\r\n", "\n").split("\n\n")]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip("\n"))
        buf = ""

    for block in blocks:
        if len(block) > target_chars:
            flush()
            # split the oversized block on line boundaries
            line_buf = ""
            for line in block.split("\n"):
                if line_buf and len(line_buf) + len(line) + 1 > target_chars:
                    chunks.append(line_buf.strip("\n"))
                    line_buf = ""
                line_buf += (("\n" if line_buf else "") + line)
            if line_buf.strip():
                chunks.append(line_buf.strip("\n"))
            continue
        if buf and len(buf) + len(block) + 2 > target_chars:
            flush()
        buf += (("\n\n" if buf else "") + block)
    flush()
    return [c for c in chunks if c.strip()]


def ingest_document(store: VerbatimStore, session_id: str, path: str, *,
                    role: str = "memory", target_chars: int = 700) -> int:
    """Ingest one document into the durable namespace, chunked + content-hash deduped.

    Returns the number of NEW chunks stored (0 if already fully ingested / file missing)."""
    try:
        text = _read(path)
    except Exception:
        return 0
    if not text:
        return 0
    seen = store.content_hashes(session_id)
    added = 0
    for chunk in chunk_document(text, target_chars):
        h = store.hash_content(chunk)
        if h in seen:
            continue
        turn = store.max_turn(session_id) + 1
        store.add_message(session_id, turn, role, chunk, token_count=None)
        seen.add(h)
        added += 1
    return added


def ensure_loaded(store: VerbatimStore, cfg: Any) -> dict:
    """Idempotently ingest every configured durable source into ``cfg.durable_session``.

    ``cfg.durable_sources`` is a list of ``{"path": ..., "role": ...}`` (role optional, default
    ``"memory"``). Returns {path: chunks_added}. Safe to call on every construction."""
    out: dict = {}
    if not getattr(cfg, "durable_memory", False):
        return out
    sources = getattr(cfg, "durable_sources", None) or []
    sid = getattr(cfg, "durable_session", "__cmx_durable__")
    target = int(getattr(cfg, "durable_chunk_chars", 700) or 700)
    for src in sources:
        if isinstance(src, str):
            path, role = src, "memory"
        elif isinstance(src, dict):
            path, role = src.get("path", ""), src.get("role", "memory")
        else:
            continue
        if not path:
            continue
        out[path] = ingest_document(store, sid, os.path.expanduser(path),
                                    role=role, target_chars=target)
    return out


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()
