"""Every-turn live-conversation retrieval (the 25+ turn truncation fix).

Regression for the multi-turn truncation finding: cmx's conversation retrieval ran only inside
compress() (fires at 75% of window = 300K for gpt-5-mini), so a normal multi-turn chat never
triggered it and providers that truncate raw history server-side dropped early turns. The
pre_llm_call conversation_context() hook injects the relevant stored turns every turn.
"""
import tempfile

from cmx.config import CmxConfig
from cmx.hermes_engine import CmxContextEngine


def _engine(tmp_path):
    cfg = CmxConfig()
    cfg.database_path = str(tmp_path / "cmx.db")
    cfg.use_embeddings = False
    cfg.rerank = False
    eng = CmxContextEngine(config=cfg)
    eng.on_session_start("canary", model="gpt-5-mini")
    return eng


def _long_chat():
    msgs = [{"role": "system", "content": "assistant"}]
    facts = [("My lucky number is 4827.", "Noted."),
             ("My project is called Polaris.", "Got it."),
             ("I live in Amsterdam, city of canals and tulips.", "Lovely."),
             ("My cat is named Mochi.", "Mochi!")]
    for u, a in facts:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    for t in range(25):
        msgs += [{"role": "user", "content": f"filler about weather {t}"},
                 {"role": "assistant", "content": f"sure {t}"}]
    return msgs


def test_compress_never_fires_on_small_multiturn_chat(tmp_path):
    """Root cause: a ~23K-token chat never reaches the 300K compaction threshold."""
    eng = _engine(tmp_path)
    assert eng.should_compress(23000) is False


def test_conversation_context_injects_early_facts_without_compaction(tmp_path):
    """Each recall question must surface its early-turn fact via every-turn retrieval."""
    eng = _engine(tmp_path)
    msgs = _long_chat()
    cases = {"what is my lucky number?": "4827",
             "what is my project called?": "Polaris",
             "what city do I live in?": "Amsterdam",
             "what is my cat named?": "Mochi"}
    for q, expected in cases.items():
        block = eng.conversation_context(q, msgs)
        assert expected in block, f"{expected!r} not retrieved for {q!r}"


def test_conversation_context_ingests_when_compress_never_ran(tmp_path):
    """The hook must ingest history itself, so the store is populated on chats that never
    compact or call a tool."""
    eng = _engine(tmp_path)
    assert eng.store.count("canary") == 0
    eng.conversation_context("hello?", _long_chat())
    assert eng.store.count("canary") > 0


def test_conversation_injection_can_be_disabled(tmp_path):
    eng = _engine(tmp_path)
    eng.cfg.inject_conversation = False
    assert eng.conversation_context("what is my lucky number?", _long_chat()) == ""


def test_recency_floor_surfaces_recent_turns_for_vague_query(tmp_path):
    """The amnesia fix: a low-signal continuation ('ok continue') must still inject
    the most-recent turns. Query-similarity alone scatters across history and never
    surfaces the recent context; the recency floor guarantees the tail is present.
    """
    eng = _engine(tmp_path)
    eng.cfg.conversation_recent_floor = 8
    # Build a long chat whose LAST distinctive content is a unique sentinel that has
    # ZERO lexical overlap with the vague query, so only the recency floor can surface it.
    msgs = [{"role": "system", "content": "assistant"}]
    for t in range(40):
        msgs += [{"role": "user", "content": f"filler topic alpha {t}"},
                 {"role": "assistant", "content": f"ack {t}"}]
    msgs += [{"role": "user", "content": "Remember the deploy passphrase is ZIRCON-RABBIT-91."},
             {"role": "assistant", "content": "Stored the passphrase."}]
    # A vague continuation with no overlap with 'ZIRCON' or 'passphrase'.
    block = eng.conversation_context("ok cool, continue", msgs)
    assert "ZIRCON-RABBIT-91" in block, "recent sentinel not injected by the recency floor"


def test_recency_floor_disabled_when_zero(tmp_path):
    """floor=0 reverts to pure query-similarity (no recent guarantee)."""
    eng = _engine(tmp_path)
    eng.cfg.conversation_recent_floor = 0
    msgs = [{"role": "system", "content": "assistant"}]
    for t in range(40):
        msgs += [{"role": "user", "content": f"filler topic beta {t}"},
                 {"role": "assistant", "content": f"ok {t}"}]
    msgs += [{"role": "user", "content": "The widget id is QUARTZ-7."},
             {"role": "assistant", "content": "noted"}]
    block = eng.conversation_context("ok continue", msgs)
    # With the floor off, the unique recent sentinel is NOT guaranteed (vague query
    # has no overlap with 'QUARTZ'); this documents the pre-fix behavior.
    assert "QUARTZ-7" not in block
