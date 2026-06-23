from cmx.store import VerbatimStore


def _store(tmp_path):
    return VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)


def test_append_only_and_recover(tmp_path):
    s = _store(tmp_path)
    i1 = s.add_message("sess", 0, "user", "We decided to use PostgreSQL for the dashboard.")
    i2 = s.add_message("sess", 1, "assistant", "Acknowledged: PostgreSQL chosen.")
    assert s.count("sess") == 2
    # verbatim recovered exactly
    assert s.get_message(i1)["content"] == "We decided to use PostgreSQL for the dashboard."
    assert s.get_message(i2)["role"] == "assistant"


def test_recent_and_pin(tmp_path):
    s = _store(tmp_path)
    for t in range(10):
        s.add_message("sess", t, "user", f"turn {t}")
    recent = s.recent("sess", 3)
    assert [m["turn_index"] for m in recent] == [7, 8, 9]
    pid = s.add_message("sess", 99, "system", "IDENTITY", pinned=True)
    assert s.get_message(pid)["pinned"] == 1
    assert any(m["id"] == pid for m in s.pinned("sess"))


def test_fts_word_search(tmp_path):
    s = _store(tmp_path)
    s.add_message("sess", 0, "user", "The migration runbook lives in the ops wiki.")
    s.add_message("sess", 1, "user", "Unrelated chatter about lunch.")
    hits = s.search_fts("migration runbook", k=5, session_id="sess")
    assert hits and hits[0]["id"] == 1


def test_trigram_finds_identifier_substring(tmp_path):
    # The decisive trigram win: an identifier that word-tokenizers split/miss,
    # found by substring.
    s = _store(tmp_path)
    fid = s.add_message("sess", 0, "assistant", "Patched job CI4_migrate_v2 in the pipeline.")
    s.add_message("sess", 1, "assistant", "Other noise about the weather.")
    assert s.has_trigram is True
    trgm = s.search_trgm("migrate_v2", k=5, session_id="sess")
    assert trgm and trgm[0]["id"] == fid


def test_session_isolation_in_search(tmp_path):
    s = _store(tmp_path)
    s.add_message("a", 0, "user", "alpha secret token zzz")
    s.add_message("b", 0, "user", "alpha secret token zzz")
    hits = s.search_fts("secret token", k=10, session_id="a")
    assert all(s.get_message(h["id"])["session_id"] == "a" for h in hits)
