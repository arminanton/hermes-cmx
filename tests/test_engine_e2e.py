"""End-to-end proof: the engine makes grounding independent of model behavior.

A *guessing* model must be caught and refused; a *phantom-citing* model must be
caught; a model that ignores injected evidence must be *forced* to retrieve. None
of these depend on the model cooperating.
"""
from cmx.config import CmxConfig
from cmx.engine import REFUSAL, CmxEngine
from cmx.llm import Completion, MockModel, ToolCall
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore


def _engine(tmp_path, model_client, verifier_client=None, cfg=None):
    cfg = cfg or CmxConfig()
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    r = HybridRetriever(s, cfg)
    return s, CmxEngine(cfg, s, r, model_client, verifier_client)


def _seed(store, fact, session="sess"):
    for t in range(30):
        store.add_message(session, t, "user", f"smalltalk number {t}")
    fid = store.add_message(session, 30, "assistant", fact)
    for t in range(31, 50):
        store.add_message(session, t, "user", f"smalltalk number {t}")
    return fid


def test_grounded_model_passes_first_try(tmp_path):
    s, _ = _engine(tmp_path, MockModel(["placeholder"]))
    fid = _seed(s, "Decision: the deploy region is eu-west-3.")
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()),
                    MockModel([f"We deploy to eu-west-3 [id={fid}]."]))
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.grounded and not resp.refused and resp.attempts == 1
    assert fid in resp.evidence_ids


def test_guessing_model_is_refused_not_shipped(tmp_path):
    # THE核心 proof: a model that asserts an unsupported fact, repeatedly, is REFUSED.
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    fid = _seed(s, "Decision: the deploy region is eu-west-3.")
    guesser = MockModel(["We deploy to eu-west-9."] * 5)   # confident, wrong, uncited
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()), guesser,
                    verifier_client=None)  # no judge → deterministic-only → safe
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.refused and not resp.grounded
    assert resp.text == REFUSAL
    assert "eu-west-9" not in resp.text          # the hallucination never shipped


def test_guess_then_correct_recovers_via_regeneration(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    fid = _seed(s, "Decision: the deploy region is eu-west-3.")
    model = MockModel([
        "We deploy to eu-west-9.",                       # attempt 1: wrong, uncited → caught
        f"Correction: we deploy to eu-west-3 [id={fid}].",  # attempt 2: grounded
    ])
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()), model)
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.grounded and resp.attempts == 2


def test_phantom_citation_is_refused(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    _seed(s, "Decision: the deploy region is eu-west-3.")
    model = MockModel(["We deploy to eu-west-3 [id=99999]."] * 5)   # cites a non-existent row
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()), model)
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.refused
    assert any("99999" in f for f in resp.failures)


def test_forced_gate_makes_model_retrieve_when_injection_misses(tmp_path):
    # Fact text is disjoint from the question → proactive injection finds nothing →
    # the forced gate must compel a retrieval before any answer.
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    fid = s.add_message("sess", 0, "assistant", "Token: aaa-bbb-ccc-111.")
    for t in range(1, 20):
        s.add_message("sess", t, "user", f"unrelated chatter {t}")

    def behavior(messages, tool_choice):
        if tool_choice == "required":
            return Completion(tool_calls=[ToolCall("cmx_grep", {"query": "Token"})])
        return Completion(content=f"The token is aaa-bbb-ccc-111 [id={fid}].")

    model = MockModel([behavior, behavior])
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()), model)
    resp = eng.respond("sess", "remind me what we decided earlier?", model="gpt-5-mini")

    assert model.calls[0]["tool_choice"] == "required"   # the gate fired
    assert resp.grounded                                 # and grounding succeeded via retrieval


def test_non_factual_turn_skips_enforcement(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    s.add_message("sess", 0, "user", "hello")
    model = MockModel(["Here is a haiku about the sea: waves crash, gulls cry, salt."])
    eng = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()), model)
    resp = eng.respond("sess", "write me a haiku about the sea", model="gpt-5.5")
    assert resp.grounded and resp.attempts == 1   # creative turn ships freely


# -- lever 6: semantic rescue (paraphrase ships; phantom still refused) -----

def test_lever6_semantic_rescue_ships_paraphrase(tmp_path):
    # evidence states the region in words; the model answers with the equivalent code,
    # so the deterministic fact-token check fails. Without the semantic verifier this is
    # refused (false refusal); with it, an independent SUPPORTED judge rescues it.
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    fid = _seed(s, "Decision: deploy region is the Paris datacenter.")
    ans = f"We deploy to eu-west-3 [id={fid}]."

    off = CmxEngine(CmxConfig(), s, HybridRetriever(s, CmxConfig()),
                    MockModel([ans] * 4), verifier_client=MockModel(["SUPPORTED"] * 4))
    r_off = off.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert r_off.refused  # strict deterministic check alone rejects the paraphrase

    cfg = CmxConfig(); cfg.semantic_verifier = True
    on = CmxEngine(cfg, s, HybridRetriever(s, cfg),
                   MockModel([ans] * 4), verifier_client=MockModel(["SUPPORTED"] * 4))
    r_on = on.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert r_on.grounded and not r_on.refused and f"[id={fid}]" in r_on.text


def test_lever6_never_rescues_phantom_even_with_yes_judge(tmp_path):
    cfg = CmxConfig(); cfg.semantic_verifier = True
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    _seed(s, "Decision: deploy region is the Paris datacenter.")
    # cites a non-existent row; even a rubber-stamp judge must NOT rescue a phantom id
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg),
                    MockModel(["We deploy to eu-west-3 [id=99999]."] * 4),
                    verifier_client=MockModel(["SUPPORTED"] * 4))
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.refused and "eu-west-3" not in resp.text


# -- iter3 L2: evidence-sufficiency pre-gate -------------------------------

def test_L2_sufficiency_gate_refuses_when_judge_says_insufficient(tmp_path):
    cfg = CmxConfig(); cfg.sufficiency_gate = True
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    _seed(s, "Decision: the deploy region is eu-west-3.")
    # the foreground WOULD answer, but the sufficiency judge says NO → refuse before generating
    fg = MockModel(["We deploy to eu-west-3 [id=31]."] * 4)
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), fg, verifier_client=MockModel(["NO"]))
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.refused and resp.text == REFUSAL
    assert fg.calls == []  # foreground never invoked — no confabulation opportunity


def test_L2_sufficiency_gate_allows_when_judge_says_answerable(tmp_path):
    cfg = CmxConfig(); cfg.sufficiency_gate = True
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    fid = _seed(s, "Decision: the deploy region is eu-west-3.")
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg),
                    MockModel([f"We deploy to eu-west-3 [id={fid}]."]),
                    verifier_client=MockModel(["YES"]))
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.grounded and not resp.refused and fid in resp.evidence_ids


# -- iter3 L5: self-consistency refusal ------------------------------------

def test_L5_self_consistency_refuses_when_fact_drifts(tmp_path):
    cfg = CmxConfig(); cfg.self_consistency = 2
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    fid = _seed(s, "Decision: the deploy region is eu-west-3.")
    # primary cites correctly (passes checks), but resamples drift to other values → refuse
    model = MockModel([f"We deploy to eu-west-3 [id={fid}].",
                       "We deploy to eu-west-9.", "We deploy to us-east-1."])
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), model)
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.refused and resp.text == REFUSAL


def test_L5_self_consistency_ships_when_fact_stable(tmp_path):
    cfg = CmxConfig(); cfg.self_consistency = 2
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    fid = _seed(s, "Decision: the deploy region is eu-west-3.")
    ans = f"We deploy to eu-west-3 [id={fid}]."
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), MockModel([ans, ans, ans]))
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.grounded and not resp.refused


def test_refusal_mode_caveat_keeps_answer_instead_of_deleting_it(tmp_path):
    """Non-destructive degrade: when a grounding gate fails, refusal_mode='caveat' ships the
    model's real answer WITH a warning appended — it must not be replaced by the canned REFUSAL.
    (The opus-4.8 fix: a misfiring gate should never delete a substantive answer.)"""
    cfg = CmxConfig(); cfg.refusal_mode = "caveat"
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    _seed(s, "Decision: the deploy region is eu-west-3.")
    guesser = MockModel(["We deploy to eu-west-9."] * 5)  # ungrounded, would normally REFUSE
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), guesser, verifier_client=None)
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.text != REFUSAL
    assert "eu-west-9" in resp.text          # the model's real answer is preserved
    assert "[cmx:" in resp.text              # with a non-destructive caveat
    assert not resp.refused


def test_refusal_mode_off_ships_answer_verbatim(tmp_path):
    """refusal_mode='off' ships the model's answer as-is (no REFUSAL, no caveat)."""
    cfg = CmxConfig(); cfg.refusal_mode = "off"
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    _seed(s, "Decision: the deploy region is eu-west-3.")
    guesser = MockModel(["We deploy to eu-west-9."] * 5)
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), guesser, verifier_client=None)
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.text == "We deploy to eu-west-9." and not resp.refused


def test_refusal_mode_replace_is_unchanged_default(tmp_path):
    """Default refusal_mode='replace' preserves the strict/benchmark behavior: REFUSAL ships."""
    cfg = CmxConfig()  # default
    assert cfg.refusal_mode == "replace"
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    _seed(s, "Decision: the deploy region is eu-west-3.")
    guesser = MockModel(["We deploy to eu-west-9."] * 5)
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), guesser, verifier_client=None)
    resp = eng.respond("sess", "which deploy region did we choose?", model="gpt-5-mini")
    assert resp.refused and resp.text == REFUSAL and "eu-west-9" not in resp.text
