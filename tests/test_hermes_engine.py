from cmx.config import CmxConfig
from cmx.hermes_engine import CmxContextEngine


def _engine(tmp_path):
    cfg = CmxConfig()
    cfg.database_path = str(tmp_path / "cmx.db")
    cfg.models = {"gpt-5-mini": {"window": 128000}}
    e = CmxContextEngine(config=cfg)
    e.on_session_start("sess", model="gpt-5-mini")
    return e


def test_identity_and_status(tmp_path):
    e = _engine(tmp_path)
    assert e.name == "cmx"
    st = e.get_status()
    assert st["engine"] == "cmx" and st["trigram"] in (True, False)


def test_should_compress_threshold(tmp_path):
    e = _engine(tmp_path)
    e.threshold_tokens = 1000
    assert e.should_compress(1500) is True
    assert e.should_compress(500) is False


def test_compress_injects_evidence_not_summary(tmp_path):
    e = _engine(tmp_path)
    msgs = [{"role": "system", "content": "SYS"}]
    for t in range(40):
        msgs.append({"role": "user", "content": f"chitchat {t}"})
    msgs.append({"role": "assistant", "content": "Decision: the cache TTL is 300 seconds."})
    for t in range(40, 55):
        msgs.append({"role": "user", "content": f"chitchat {t}"})
    msgs.append({"role": "user", "content": "what cache TTL did we set?"})
    out = e.compress(msgs)
    sys_block = out[0]["content"]
    assert "CMX EVIDENCE" in sys_block and "300 seconds" in sys_block  # injected verbatim
    # never a lossy summary, never assistant-role injected history
    assert "[Recent Summary" not in sys_block


def test_tools_served(tmp_path):
    e = _engine(tmp_path)
    e.compress([{"role": "assistant", "content": "The release tag is v4.2.0-rc7."},
                {"role": "user", "content": "what tag?"}])
    import json
    grep = json.loads(e.handle_tool_call("cmx_grep", {"query": "v4.2.0-rc7"}))
    assert grep["matches"] and "v4.2.0-rc7" in grep["matches"][0]["content"]
    mid = grep["matches"][0]["id"]
    exp = json.loads(e.handle_tool_call("cmx_expand", {"id": mid}))
    assert "v4.2.0-rc7" in exp["content"]
    rec = json.loads(e.handle_tool_call("cmx_recall", {"n": 5}))
    assert "turns" in rec


def test_reset_never_purges_verbatim(tmp_path):
    e = _engine(tmp_path)
    e.compress([{"role": "user", "content": "fact zeta-9 persists"}])
    before = e.store.count("sess")
    e.on_session_reset()
    assert e.store.count("sess") == before  # lossless across /new


def _enforce_msgs(e):
    fid = e.store.add_message("sess", 0, "assistant", "Decision: the deploy region is eu-west-3.")
    msgs = [{"role": "system",
             "content": f"[CMX EVIDENCE] [id={fid}] (assistant): Decision: the deploy region is eu-west-3."},
            {"role": "user", "content": "which deploy region did we decide on?"}]
    return fid, msgs


def test_enforce_capability_and_grounded_accept(tmp_path):
    e = _engine(tmp_path)
    fid, msgs = _enforce_msgs(e)
    assert e.capabilities().get("enforce_response") is True
    assert e.enforce_response(f"We deploy to eu-west-3 [id={fid}].", msgs)["action"] == "accept"


def test_enforce_phantom_citation_regenerates(tmp_path):
    e = _engine(tmp_path)
    _, msgs = _enforce_msgs(e)
    out = e.enforce_response("We deploy to eu-west-3 [id=99999].", msgs)
    assert out["action"] == "regenerate" and "99999" in out["message"]


def test_enforce_fabrication_regenerates_then_refuses(tmp_path):
    e = _engine(tmp_path)
    _, msgs = _enforce_msgs(e)
    # uncited fabricated value not present in the assembled verbatim
    assert e.enforce_response("We deploy to eu-west-9.", msgs)["action"] == "regenerate"
    final = e.enforce_response("We deploy to eu-west-9.", msgs, final=True)
    assert final["action"] == "replace" and "record" in final["text"].lower()


def test_enforce_skips_non_factual(tmp_path):
    e = _engine(tmp_path)
    msgs = [{"role": "user", "content": "write me a haiku about the sea"}]
    assert e.enforce_response("waves crash, gulls cry, salt air.", msgs)["action"] == "accept"


def test_enforce_refusal_mode_off_disables_capability_and_never_replaces(tmp_path):
    """refusal_mode='off' is the master switch: capability is withdrawn (host won't call it)
    AND a direct call accepts the answer instead of deleting it."""
    e = _engine(tmp_path)
    e.cfg.refusal_mode = "off"
    _, msgs = _enforce_msgs(e)
    assert e.capabilities().get("enforce_response") is False     # host skips enforcement entirely
    out = e.enforce_response("We deploy to eu-west-9.", msgs, final=True)
    assert out["action"] == "accept"                              # ungrounded answer NOT nuked


def test_enforce_refusal_mode_caveat_keeps_answer_with_warning(tmp_path):
    """refusal_mode='caveat' keeps the model's real answer + a warning instead of REFUSAL."""
    e = _engine(tmp_path)
    e.cfg.refusal_mode = "caveat"
    _, msgs = _enforce_msgs(e)
    assert e.capabilities().get("enforce_response") is True       # still advertised
    final = e.enforce_response("We deploy to eu-west-9.", msgs, final=True)
    assert final["action"] == "replace"
    assert "eu-west-9" in final["text"]                           # the real answer survives
    assert "[cmx:" in final["text"]                               # with a non-destructive caveat
    assert "record" not in final["text"].lower().split("[cmx:")[0]  # not the bare REFUSAL


def test_enforce_refusal_mode_replace_is_default_and_unchanged(tmp_path):
    """Default refusal_mode='replace' preserves the strict REFUSAL behavior."""
    e = _engine(tmp_path)
    assert e.cfg.refusal_mode == "replace"
    _, msgs = _enforce_msgs(e)
    final = e.enforce_response("We deploy to eu-west-9.", msgs, final=True)
    assert final["action"] == "replace" and "record" in final["text"].lower()
    assert "eu-west-9" not in final["text"]


def test_copilot_claude_window_is_1m_not_200k(tmp_path):
    """Regression: cmx must treat Copilot Claude opus-4.8 as a 1M window, not the
    vendor-base 200k. The TUI reads comp.context_length directly, so a 200k here
    caused the recurring "0/200k" display + premature compaction at ~150k."""
    from cmx.tokenizer import window_for
    assert window_for("claude-opus-4.8") == 1_000_000
    assert window_for("claude-opus-4.7") == 1_000_000

    e = _engine(tmp_path)
    e.on_session_start("sess", model="claude-opus-4.8")
    assert e.context_length == 1_000_000


def test_on_session_start_preserves_hermes_resolved_window(tmp_path):
    """Regression: on_session_start (which fires on every compaction rollover) must
    NOT clobber the live window that update_model resolved. Previously it re-derived
    from the static table and snapped 1M back to 200k on the next rollover."""
    e = _engine(tmp_path)
    # Hermes resolves the authoritative window for a model the static table omits.
    e.update_model(model="some-future-claude", context_length=1_000_000)
    assert e.context_length == 1_000_000
    # A rollover / fresh session_start for the same model must keep 1M.
    e.on_session_start("sess2", model="some-future-claude")
    assert e.context_length == 1_000_000
    # compaction-rollover variant carries old_session_id; still preserved.
    e.on_session_start("sess3", model="some-future-claude",
                       boundary_reason="compression", old_session_id="sess2")
    assert e.context_length == 1_000_000
