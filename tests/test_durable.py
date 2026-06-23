"""Stage 1 durable-memory tests: verbatim ingest + retrieval + no-drop budgeting."""
from cmx.config import CmxConfig
from cmx.durable import chunk_document, ensure_loaded, ingest_document
from cmx.hermes_engine import CmxContextEngine
from cmx.store import VerbatimStore


def _store(tmp_path):
    return VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)


def test_chunk_document_is_verbatim_and_segmented():
    text = "# Title\n\nAlpha fact about the dashboard.\n\n" + ("x " * 500) + "\n\nBravo fact."
    chunks = chunk_document(text, target_chars=200)
    assert len(chunks) >= 2
    # verbatim: every chunk's text appears in the source
    for c in chunks:
        assert c.strip() in text or c.strip().replace("\n", " ") in text.replace("\n", " ")
    joined = " ".join(chunks)
    assert "Alpha fact" in joined and "Bravo fact" in joined


def test_ingest_is_idempotent(tmp_path):
    s = _store(tmp_path)
    doc = tmp_path / "MEMORY.md"
    doc.write_text("Para one about postgres.\n\nPara two about redis.\n\nPara three about kafka.")
    n1 = ingest_document(s, "__dur__", str(doc), role="memory", target_chars=80)
    n2 = ingest_document(s, "__dur__", str(doc), role="memory", target_chars=80)
    assert n1 >= 1
    assert n2 == 0  # nothing new on re-ingest (content-hash dedup)
    assert s.count("__dur__") == n1


def test_durable_memory_retrieved_and_searchable(tmp_path):
    doc = tmp_path / "MEMORY.md"
    doc.write_text(
        "The production deploy region is eu-west-3.\n\n"
        "The on-call rotation uses PagerDuty schedule P12.\n\n"
        "The billing owner is Dana in finance.")
    cfg = CmxConfig()
    cfg.database_path = str(tmp_path / "cmx.db")
    cfg.durable_memory = True
    cfg.durable_session = "__dur__"
    cfg.durable_sources = [{"path": str(doc), "role": "memory"}]
    cfg.durable_chunk_chars = 120

    eng = CmxContextEngine(config=cfg)  # __init__ auto-loads durable memory
    assert eng.store.count("__dur__") >= 2  # chunked (small paras may merge under target)

    # a conversation turn whose answer lives ONLY in durable memory
    eng.on_session_start("conv-1", model="gpt-5-mini")
    msgs = [{"role": "system", "content": "assistant"},
            {"role": "user", "content": "which production deploy region did we pick?"}]
    assembled = eng.compress(msgs)
    sys_block = assembled[0]["content"]
    assert "DURABLE MEMORY" in sys_block
    assert "eu-west-3" in sys_block  # retrieved from durable, not dumped wholesale

    # and it's reachable on demand via cmx_grep (the no-drop guarantee)
    import json
    res = json.loads(eng.handle_tool_call("cmx_grep", {"query": "billing owner"}))
    assert any("Dana" in m.get("content", "") for m in res["matches"])


def test_durable_off_is_noop(tmp_path):
    cfg = CmxConfig()
    cfg.database_path = str(tmp_path / "cmx.db")
    cfg.durable_memory = False
    cfg.durable_sources = [{"path": str(tmp_path / "nope.md")}]
    eng = CmxContextEngine(config=cfg)
    eng.on_session_start("c", model="gpt-5-mini")
    out = eng.compress([{"role": "user", "content": "hello"}])
    # no durable block when disabled
    assert not any("DURABLE MEMORY" in (m.get("content") or "") for m in out)


def test_durable_context_injects_every_turn(tmp_path):
    """durable_context() (the pre_llm_call every-turn path) returns the relevant durable
    slice WITHOUT needing compaction — fixes the 'only injected on compaction' bug."""
    doc = tmp_path / "USER.md"
    doc.write_text(
        "The user's preferred TTS voice is British male, set on 2026-06-10.\n\n"
        "The user's editor is neovim.\n\n"
        "The user's timezone is America/Vancouver.")
    cfg = CmxConfig()
    cfg.database_path = str(tmp_path / "cmx.db")
    cfg.durable_memory = True
    cfg.durable_session = "__dur__"
    cfg.durable_sources = [{"path": str(doc), "role": "user"}]
    cfg.durable_chunk_chars = 80
    eng = CmxContextEngine(config=cfg)

    # NO compaction / compress() call — just the every-turn hook path
    block = eng.durable_context("what TTS voice did I pick?")
    assert "DURABLE MEMORY" in block
    assert "British male" in block  # the exact fact, retrieved namespaced + bm25-ranked

    # disabled → empty
    cfg.durable_memory = False
    assert eng.durable_context("what TTS voice did I pick?") == ""
