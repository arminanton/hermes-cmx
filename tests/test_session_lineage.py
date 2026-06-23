"""Tests for session lineage across Hermes compaction rollover (the session_id-rotation bug).

Hermes rotates session_id on compaction and notifies the engine via on_session_start(new,
boundary_reason="compression", old_session_id=old). Without lineage, all session-scoped
retrieval would lose the prior conversation. These prove the root-normalization fix.
"""
from cmx.config import CmxConfig
from cmx.hermes_engine import CmxContextEngine
from cmx.store import VerbatimStore


def _store(tmp_path):
    return VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)


# -- store-level lineage ------------------------------------------------------

def test_root_session_standalone_is_self(tmp_path):
    s = _store(tmp_path)
    assert s.root_session("sess-A") == "sess-A"


def test_link_session_resolves_to_root(tmp_path):
    s = _store(tmp_path)
    # A -> B (first compaction): B's root is A
    assert s.link_session("B", "A") == "A"
    assert s.root_session("B") == "A"
    assert s.root_session("A") == "A"
    # B -> C (second compaction): C's root is still A
    assert s.link_session("C", "B") == "A"
    assert s.root_session("C") == "A"


def test_lineage_persists_across_store_reopen(tmp_path):
    s = _store(tmp_path)
    s.link_session("B", "A")
    s.conn.commit() if hasattr(s.conn, "commit") else None
    # reopen the same db file → a resumed child must still resolve to its root
    s2 = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    assert s2.root_session("B") == "A"


# -- engine-level: the actual bug scenario ------------------------------------

def _engine(tmp_path):
    cfg = CmxConfig(); cfg.use_embeddings = False; cfg.rerank = False
    eng = CmxContextEngine(config=cfg)
    # point the engine's store at a temp db
    eng.store = _store(tmp_path)
    from cmx.retrieval import HybridRetriever
    eng.retriever = HybridRetriever(eng.store, cfg)
    return eng


def test_retrieval_survives_compaction_rollover(tmp_path):
    eng = _engine(tmp_path)
    eng.on_session_start("conv-A", model="gpt-5-mini")
    # store a fact in the original session
    eng.store.add_message(eng.session_id, 0, "user", "the deploy port is 8080")
    assert eng.session_id == "conv-A"

    # Hermes compacts → rotates session id, notifying the engine
    eng.on_session_start("conv-B", boundary_reason="compression", old_session_id="conv-A",
                         model="gpt-5-mini")
    # effective session stays the ROOT, so the prior fact is still retrievable
    assert eng.session_id == "conv-A"
    hits = eng.retriever.retrieve("what deploy port?", session_id=eng.session_id)
    assert any("8080" in h.content for h in hits), "prior conversation lost after rollover!"

    # a new turn after the rollover lands under the same root
    eng.store.add_message(eng.session_id, eng.store.max_turn(eng.session_id) + 1,
                          "user", "the region is eu-west-3")
    assert eng.store.count("conv-A") == 2


def test_resume_child_session_finds_history(tmp_path):
    eng = _engine(tmp_path)
    eng.on_session_start("conv-A", model="gpt-5-mini")
    eng.store.add_message(eng.session_id, 0, "user", "the api key is sk-test-123")
    eng.on_session_start("conv-B", boundary_reason="compression", old_session_id="conv-A")
    # simulate a later process resuming the CHILD session id directly (no boundary signal)
    eng2 = _engine_sharing_db(tmp_path)
    eng2.on_session_start("conv-B", model="gpt-5-mini")   # resume child
    assert eng2.session_id == "conv-A"                     # resolved to root
    hits = eng2.retriever.retrieve("api key?", session_id=eng2.session_id)
    assert any("sk-test-123" in h.content for h in hits)


def _engine_sharing_db(tmp_path):
    cfg = CmxConfig(); cfg.use_embeddings = False; cfg.rerank = False
    eng = CmxContextEngine(config=cfg)
    eng.store = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    from cmx.retrieval import HybridRetriever
    eng.retriever = HybridRetriever(eng.store, cfg)
    return eng


def test_no_lineage_signal_is_standalone(tmp_path):
    eng = _engine(tmp_path)
    eng.on_session_start("solo", model="gpt-5-mini")
    assert eng.session_id == "solo"  # no rollover → unchanged


# -- ingest robustness across compaction (the positional-cursor bug) ----------

def test_ingest_dedups_and_survives_compaction(tmp_path):
    eng = _engine(tmp_path)
    eng.on_session_start("conv-A", model="gpt-5-mini")
    # first compaction sees the full early history
    eng._ingest_new([
        {"role": "user", "content": "the port is 8080"},
        {"role": "assistant", "content": "noted, port 8080"},
    ])
    assert eng.store.count("conv-A") == 2
    # re-feeding the SAME messages (host replays the working set) must NOT duplicate
    eng._ingest_new([
        {"role": "user", "content": "the port is 8080"},
        {"role": "assistant", "content": "noted, port 8080"},
        {"role": "user", "content": "the region is eu-west-3"},   # one genuinely new turn
    ])
    assert eng.store.count("conv-A") == 3  # only the new turn added, no dupes

    # after a rollover, a brand-new turn still lands (cursor would have broken here)
    eng.on_session_start("conv-B", boundary_reason="compression", old_session_id="conv-A")
    eng._ingest_new([{"role": "user", "content": "the db is postgres"}])
    assert eng.store.count("conv-A") == 4
    hits = eng.retriever.retrieve("which db?", session_id=eng.session_id)
    assert any("postgres" in h.content for h in hits)


def test_on_session_end_flushes_short_session(tmp_path):
    eng = _engine(tmp_path)
    eng.on_session_start("short", model="gpt-5-mini")
    # a short session that never compacted — on_session_end must still capture it
    eng.on_session_end("short", [
        {"role": "user", "content": "my favorite number is 42"},
        {"role": "assistant", "content": "got it, 42"},
    ])
    assert eng.store.count("short") == 2
