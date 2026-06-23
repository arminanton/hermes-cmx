"""Example 1 — verbatim storage + hybrid retrieval (no model needed, runs instantly).

Shows the core of cmx: every message is stored *verbatim* in SQLite, and the engine
retrieves the exact relevant slice on demand — even when it is buried in noise. This is
the opposite of LCM, which would have replaced the old turns with a lossy summary.

Run:
    PYTHONPATH=src python3 examples/01_store_and_retrieve.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmx.config import CmxConfig
from cmx.hermes_engine import CmxContextEngine


def main():
    cfg = CmxConfig()
    cfg.database_path = tempfile.mktemp(suffix=".db")
    engine = CmxContextEngine(config=cfg)
    engine.on_session_start("demo-session", model="gpt-5-mini")
    sid = engine.session_id

    # ingest a long conversation with one important fact buried in noise
    turn = 0
    for i in range(40):
        engine.store.add_message(sid, turn, "user", f"misc note {i} about unrelated chores")
        turn += 1
    engine.store.add_message(
        sid, turn, "assistant",
        "Decision: the production database is Postgres 16 in region eu-west-3.")
    turn += 1
    for i in range(40, 80):
        engine.store.add_message(sid, turn, "user", f"misc note {i} about unrelated chores")
        turn += 1

    print(f"stored {engine.store.count(sid)} verbatim messages (nothing summarized away)\n")

    # retrieve the buried fact — hybrid FTS5 + trigram (+ embeddings if configured)
    query = "which production database and region did we decide on?"
    slices = engine.retriever.retrieve(query, session_id=sid, k=3)
    print(f"query: {query!r}\ntop retrieved verbatim slices:")
    for s in slices:
        print(f"  [id={s.id}] {s.content!r}")

    joined = " ".join(s.content for s in slices)
    assert "eu-west-3" in joined and "Postgres 16" in joined, "buried fact was not retrieved"
    print("\n[ok] the exact fact was recovered verbatim from 80 turns of noise ✅")


if __name__ == "__main__":
    main()
