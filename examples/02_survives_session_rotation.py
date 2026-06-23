"""Example 2 — memory survives Hermes session-id rotation (no model needed, runs instantly).

This is the bug the user caught: Hermes rotates the session_id during compaction and
checkpoints, so a naive engine that scopes retrieval by session_id loses the whole
conversation the instant it compacts. cmx normalizes every rotated id to a lineage ROOT,
so the one logical conversation keeps a single effective id across any number of rollovers.

Here we ingest under session "A", simulate a compaction rollover to "B" (exactly how the
host notifies the engine), and show the fact stored under "A" is still retrievable under "B".

Run:
    PYTHONPATH=src python3 examples/02_survives_session_rotation.py
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

    # --- session A: store a fact ---
    engine.on_session_start("session-A", model="gpt-5-mini")
    root_a = engine.session_id
    engine.store.add_message(root_a, 0, "user", "Remember: the launch code word is BLUEFERN.")
    print(f"session-A effective id = {root_a!r}; stored 1 fact")

    # --- compaction rollover: host rotates A -> B and tells the engine ---
    engine.on_session_start("session-B", boundary_reason="compression",
                            old_session_id="session-A", model="gpt-5-mini")
    root_b = engine.session_id
    print(f"after compaction, host gave new id 'session-B'; cmx normalized it to {root_b!r}")

    # --- retrieve under the NEW id ---
    slices = engine.retriever.retrieve("what is the launch code word?",
                                       session_id=engine.session_id, k=3)
    found = any("BLUEFERN" in getattr(s, "content", "") for s in slices)
    print(f"retrieving under the rotated id finds the fact from session-A: {found}")

    assert root_b == root_a, "rotated id was not normalized to the lineage root"
    assert found, "fact stored before compaction was lost after rotation"
    print("\n[ok] same logical conversation, one effective id across rotation — "
          "no memory lost on compaction/checkpoint ✅")


if __name__ == "__main__":
    main()
