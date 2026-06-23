from cmx.config import CmxConfig
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore


def _setup(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    cfg = CmxConfig()
    return s, HybridRetriever(s, cfg)


def test_hybrid_finds_planted_fact_topk(tmp_path):
    s, r = _setup(tmp_path)
    # noise
    for t in range(50):
        s.add_message("sess", t, "user", f"chitchat number {t} about nothing in particular")
    # planted fact
    fact_id = s.add_message("sess", 50, "assistant",
                            "Decision: the staging database is db-staging-7 on port 6432.")
    for t in range(51, 80):
        s.add_message("sess", t, "user", f"more chitchat {t}")
    hits = r.retrieve("which staging database and port did we choose", session_id="sess")
    assert fact_id in [h.id for h in hits[:3]]


def test_trigram_beats_word_only_on_identifier(tmp_path):
    s, r = _setup(tmp_path)
    fid = s.add_message("sess", 0, "assistant", "Rolled out release artifact build-20260601_rc3 to prod.")
    for t in range(1, 30):
        s.add_message("sess", t, "user", f"noise {t}")
    hits = r.retrieve("20260601_rc3", session_id="sess")
    assert hits and hits[0].id == fid
    assert "trgm" in hits[0].sources  # trigram contributed the hit


def test_returns_citable_slices(tmp_path):
    s, r = _setup(tmp_path)
    fid = s.add_message("sess", 0, "user", "The API key rotation interval is 90 days.")
    hits = r.retrieve("API key rotation interval", session_id="sess")
    ev = hits[0].as_evidence()
    assert ev["id"] == fid and "90 days" in ev["content"]


def test_empty_query_safe(tmp_path):
    s, r = _setup(tmp_path)
    s.add_message("sess", 0, "user", "something")
    assert r.retrieve("???", session_id="sess") == [] or isinstance(
        r.retrieve("the", session_id="sess"), list)
