"""Offline tests for the Council-backed judge (iter-4 multi-judge wiring).

The Council is mocked (monkeypatch ``cmx.council_judge._resolve_call_tool``) so these
run deterministically with no provider, no daemon, no network — they prove the WIRING:
verdict mapping, fallback-on-outage, and the engine's gate/verify seams.
"""
import cmx.council_judge as cj
from cmx.config import CmxConfig
from cmx.council_judge import CouncilVerifier, make_verifier
from cmx.engine import CmxEngine
from cmx.enforcement import Verifier
from cmx.llm import Completion, MockModel
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore


def _fake_call_tool(verdict="allow", confidence=0.8):
    """Return a fake Council ``call_tool`` that always answers ``verdict``."""
    def call_tool(name, args):
        return {"verdict": verdict, "confidence": confidence, "verdict_only": True}
    return call_tool


def _patch_council(monkeypatch, verdict="allow", confidence=0.8, raises=False):
    def resolver():
        if raises:
            raise ImportError("hermes_council not installed")
        return _fake_call_tool(verdict, confidence)
    monkeypatch.setattr(cj, "_resolve_call_tool", resolver)


# -- factory ------------------------------------------------------------------

def test_make_verifier_defaults_to_single_judge():
    cfg = CmxConfig()
    v = make_verifier(cfg, MockModel(["YES"]), "gpt-5-mini")
    assert isinstance(v, Verifier) and not isinstance(v, CouncilVerifier)


def test_make_verifier_selects_council_with_fallback():
    cfg = CmxConfig()
    cfg.judge_backend = "council"
    v = make_verifier(cfg, MockModel(["YES"]), "gpt-5-mini")
    assert isinstance(v, CouncilVerifier)
    assert isinstance(v.fallback, Verifier)  # single judge retained as fallback


def test_make_verifier_council_without_client_stays_single():
    # verifier off => no judge client => council not engaged (consistent with verifier off)
    cfg = CmxConfig()
    cfg.judge_backend = "council"
    v = make_verifier(cfg, None, "gpt-5-mini")
    assert not isinstance(v, CouncilVerifier)


# -- verdict mapping ----------------------------------------------------------

def test_answerable_allow_is_sufficient(monkeypatch):
    _patch_council(monkeypatch, verdict="allow")
    v = CouncilVerifier("opus")
    assert v.answerable("what port?", "the deploy port is 8080") is True


def test_answerable_deny_refuses(monkeypatch):
    _patch_council(monkeypatch, verdict="deny")
    v = CouncilVerifier("opus")
    assert v.answerable("what port?", "unrelated chatter about lunch") is False


def test_answerable_conditional_is_conservative_refuse(monkeypatch):
    _patch_council(monkeypatch, verdict="conditional")
    v = CouncilVerifier("opus")
    assert v.answerable("what port?", "on-topic but no port stated") is False


def test_judge_allow_supported_deny_unsupported(monkeypatch):
    _patch_council(monkeypatch, verdict="allow")
    assert CouncilVerifier("opus")._judge("port is 8080", "deploy port 8080") == "SUPPORTED"
    _patch_council(monkeypatch, verdict="deny")
    assert CouncilVerifier("opus")._judge("port is 9999", "deploy port 8080") == "UNSUPPORTED"


def test_verify_consults_council_for_uncited_claim(monkeypatch):
    # answer asserts a fact token absent from evidence => deterministic check fails =>
    # the council judge is consulted; deny => unsupported.
    _patch_council(monkeypatch, verdict="deny")
    v = CouncilVerifier("opus")
    res = v.verify("The region is eu-west-9.", "we deployed to eu-west-1 last week")
    assert res.ok is False and res.unsupported


# -- safe degradation ---------------------------------------------------------

def test_council_outage_falls_back_to_single_judge(monkeypatch):
    _patch_council(monkeypatch, raises=True)
    # fallback single judge: scripted to say the claim is SUPPORTED / question YES
    fallback = Verifier(MockModel(["YES", "SUPPORTED"]), "gpt-5-mini")
    v = CouncilVerifier("opus", fallback=fallback)
    assert v.answerable("what port?", "the deploy port is 8080") is True  # fallback YES
    assert v._judge("port 8080", "deploy port 8080") == "SUPPORTED"        # fallback SUPPORTED


def test_council_outage_no_fallback_is_safe(monkeypatch):
    _patch_council(monkeypatch, raises=True)
    v = CouncilVerifier("opus")  # no fallback
    assert v.answerable("what port?", "evidence") is True   # don't block (base safe default)
    assert v._judge("claim", "evidence") == "UNSUPPORTED"   # but never confabulate-pass


# -- engine integration (the gate seam) ---------------------------------------

def _engine(tmp_path, cfg, foreground, verifier_client):
    store = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    return CmxEngine(cfg, store, HybridRetriever(store, cfg), foreground,
                     verifier_client=verifier_client), store


def _council_cfg():
    cfg = CmxConfig()
    cfg.judge_backend = "council"
    cfg.sufficiency_gate = True
    cfg.recent_window_turns = 6
    cfg.use_embeddings = False
    cfg.rerank = False
    return cfg


def test_engine_council_gate_denies_then_refuses(monkeypatch, tmp_path):
    _patch_council(monkeypatch, verdict="deny")
    cfg = _council_cfg()
    # a guessing foreground would confabulate, but the gate must refuse first
    foreground = MockModel([Completion(content="The port is 9999.")])
    eng, store = _engine(tmp_path, cfg, foreground, MockModel(["NO"]))
    eng.ingest("s", "user", "we talked about lunch and the weather", model="gpt-5-mini")
    resp = eng.respond("s", "what port did we decide on?", model="gpt-5-mini", persist=False)
    assert resp.refused is True
    assert "record" in resp.text.lower()
    assert any("sufficiency gate" in f for f in resp.failures)


def test_engine_council_gate_allows_then_ships(monkeypatch, tmp_path):
    _patch_council(monkeypatch, verdict="allow")
    cfg = _council_cfg()
    foreground = MockModel([Completion(content="The port is 8080.")])
    eng, store = _engine(tmp_path, cfg, foreground, MockModel(["YES", "SUPPORTED"]))
    # fact lives in the recent verbatim window -> deterministic checks ground it
    eng.ingest("s", "user", "the port is 8080", model="gpt-5-mini")
    resp = eng.respond("s", "what port did we decide on?", model="gpt-5-mini", persist=False)
    assert resp.refused is False
    assert "8080" in resp.text


def test_assume_factual_history_makes_gate_fire_on_plain_question(monkeypatch, tmp_path):
    # A question with NO history-hint words is not classified factual by the heuristic,
    # so the gate would skip. assume_factual_history=True forces the gate to engage —
    # required for pure conversation-history QA benchmarks (LOCOMO/OOLONG).
    from cmx.enforcement import classify_factual_history
    plain = "Caroline's favorite color?"
    assert classify_factual_history(plain) is False        # heuristic would skip
    _patch_council(monkeypatch, verdict="deny")
    cfg = _council_cfg()
    cfg.assume_factual_history = True
    foreground = MockModel([Completion(content="blue")])
    eng, store = _engine(tmp_path, cfg, foreground, MockModel(["NO"]))
    eng.ingest("s", "user", "we chatted about the weather", model="gpt-5-mini")
    resp = eng.respond("s", plain, model="gpt-5-mini", persist=False)
    assert resp.refused is True   # gate fired (council deny) despite no history-hint words
    assert any("sufficiency gate" in f for f in resp.failures)
