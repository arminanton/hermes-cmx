"""Migration: import hermes-lcm's verbatim store into cmx.

LCM already keeps every message verbatim (its one indisputable strength), so the
import is lossless: we copy rows in store_id order, assigning a per-session
turn_index, into the cmx append-only store (which re-indexes FTS5 + trigram via
its triggers). Makes the switch reversible and zero-data-loss.
"""
from __future__ import annotations

from typing import Optional

from . import db
from .store import VerbatimStore


def import_lcm(lcm_db_path: str, cmx_store: VerbatimStore, *,
               limit: Optional[int] = None) -> dict:
    """Copy LCM messages → cmx store. Returns a summary dict."""
    src = db.connect(lcm_db_path)
    turn_by_session: dict[str, int] = {}
    imported = 0
    skipped = 0

    sql = "SELECT store_id, session_id, role, content, pinned, token_estimate " \
          "FROM messages ORDER BY store_id ASC"
    if limit:
        sql += f" LIMIT {int(limit)}"

    for r in src.execute(sql):
        session_id = r["session_id"]
        role = r["role"] or "user"
        content = r["content"]
        if content is None or content == "":
            skipped += 1
            continue
        t = turn_by_session.get(session_id, 0)
        turn_by_session[session_id] = t + 1
        cmx_store.add_message(
            session_id, t, role, content,
            pinned=bool(r["pinned"]),
            token_count=int(r["token_estimate"] or 0) or None,
        )
        imported += 1

    src.close()
    return {"imported": imported, "skipped_empty": skipped,
            "sessions": len(turn_by_session)}


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from .config import CmxConfig
    ap = argparse.ArgumentParser(prog="cmx-import-lcm")
    ap.add_argument("lcm_db")
    ap.add_argument("--cmx-db", default="")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = CmxConfig()
    store = VerbatimStore(a.cmx_db or cfg.resolved_db_path())
    print(import_lcm(a.lcm_db, store, limit=a.limit))
