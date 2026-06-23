from cmx.enforcement import (
    Verifier, check_citations, classify_factual_history, salient_fact_tokens,
    span_supported, uncited_factual_claims,
)
from cmx.llm import MockModel
from cmx.store import VerbatimStore


def test_factual_history_classifier():
    assert classify_factual_history("which database did we decide on?")
    assert classify_factual_history("what was the deploy region?")
    assert classify_factual_history("remind me of the port we chose")
    assert not classify_factual_history("write a haiku about the sea")
    assert not classify_factual_history("what is 2 + 2")  # no history hint, trivial


def test_salient_tokens_target_hallucination_surface():
    toks = set(salient_fact_tokens('the region is eu-west-3 on port 6432 named "prod-db"'))
    assert "eu-west-3" in toks and "6432" in toks and "prod-db" in toks
    # plain english not treated as a checkable fact token
    assert "region" not in toks


def test_span_supported():
    assert span_supported("region eu-west-3", "we picked eu-west-3 last week")
    assert not span_supported("region eu-west-9", "we picked eu-west-3 last week")
    assert span_supported("a soft claim with no facts", "anything")  # → defers to L4


def _store(tmp_path):
    s = VerbatimStore(str(tmp_path / "cmx.db"))
    s.add_message("sess", 0, "assistant", "Decision: deploy region is eu-west-3, port 6432.")  # id=1
    return s


def test_citation_correct_passes(tmp_path):
    s = _store(tmp_path)
    res = check_citations("We deploy to eu-west-3 on port 6432 [id=1].", s)
    assert res.ok


def test_phantom_citation_caught(tmp_path):
    s = _store(tmp_path)
    res = check_citations("We deploy to eu-west-3 [id=999].", s)
    assert not res.ok and res.failures[0].reason == "phantom_citation"


def test_citation_mismatch_caught(tmp_path):
    # cites a real row, but the fact is NOT in it → the classic confident-wrong attribution
    s = _store(tmp_path)
    res = check_citations("We deploy to eu-west-9 [id=1].", s)
    assert not res.ok and res.failures[0].reason == "citation_mismatch"


def test_uncited_factual_claims_flagged():
    ans = "The port is 6432. I think it is fine. We chose eu-west-3 [id=1]."
    claims = uncited_factual_claims(ans)
    assert any("6432" in c for c in claims)
    assert not any("[id=1]" in c for c in claims)  # cited ones excluded


def test_verifier_deterministic_support(tmp_path):
    v = Verifier(client=None, model="x")
    # evidence contains the fact → supported without any model call
    r = v.verify("The interval is 90 days.", "policy: rotation interval is 90 days")
    assert r.ok


def test_verifier_unsupported_when_absent_and_no_client():
    v = Verifier(client=None, model="x")
    r = v.verify("The interval is 45 days.", "policy: rotation interval is 90 days")
    assert not r.ok and "45 days" in r.message()


def test_verifier_uses_model_when_inconclusive():
    # claim has no exact fact-token match; a model judge decides
    judge = MockModel(["UNSUPPORTED"])
    v = Verifier(client=judge, model="gpt-5-mini")
    r = v.verify("We definitely promised a refund policy.", "discussion about shipping times")
    assert not r.ok


def test_verifier_outage_is_safe():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("verifier down")
    v = Verifier(client=Boom(), model="x")
    r = v.verify("We agreed to ship the rollout next week.", "unrelated evidence")
    assert not r.ok  # outage → unsupported → refuse, never rubber-stamp


# -- lever 6: semantic-support rescue --------------------------------------

def test_lever6_semantic_supported_requires_judge():
    # no judge available → never rescue (safe default)
    assert Verifier(client=None, model="x").semantic_supported("they hiked", "we did a trail walk") is False


def test_lever6_semantic_supported_accepts_paraphrase():
    judge = MockModel(["SUPPORTED"])
    v = Verifier(client=judge, model="gpt-5-mini")
    assert v.semantic_supported("They went hiking.", "We did a long trail walk in the hills.") is True


def test_lever6_semantic_rejects_substitution():
    # one-entity swap must be caught (the embedding-cosine failure mode the judge guards)
    judge = MockModel(["UNSUPPORTED"])
    v = Verifier(client=judge, model="gpt-5-mini")
    assert v.semantic_supported("We chose Postgres.", "We chose MySQL.") is False


def test_lever6_semantic_outage_is_safe():
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("down")
    assert Verifier(client=Boom(), model="x").semantic_supported("a", "b") is False


def test_classify_does_not_refuse_legitimate_prompts():
    """Regression: cmx is the GLOBAL engine now, so classify_factual_history must NOT flag
    instruction/persona/task prompts as recall questions — a false positive replaces a real
    answer with a refusal (the opus-4.8 persona-setup regression, 2026-06-14)."""
    from cmx.enforcement import classify_factual_history as c
    legitimate = [
        'You are a Canadian immigration law assistant. Acknowledge these instructions, then '
        'proceed with the user first question.',
        'You are a helpful assistant. Act as a senior engineer.',
        'Can you help me write a Python script to parse CSV?',
        'Should we use PostgreSQL or MySQL for this project?',
        'Please review this code and suggest improvements.',
        'Write me a poem about the ocean.',
    ]
    for t in legitimate:
        assert c(t) is False, f"false-positive recall classification on: {t!r}"
    # a long instruction block is never a recall question (length guard)
    assert c("You are an assistant. " * 60) is False


def test_classify_still_catches_genuine_recall():
    """The conservative classifier must still flag real recall questions so refuse-to-guess
    keeps working where it matters."""
    from cmx.enforcement import classify_factual_history as c
    recall = [
        'what TTS voice did I pick?',
        'what did we decide about the database schema?',
        'which AWS region did you choose for production?',
        'remember the API key I gave you earlier?',
        'what was the value of MAX_RETRIES we set?',
    ]
    for t in recall:
        assert c(t) is True, f"missed a genuine recall question: {t!r}"


def test_classify_forward_planning_is_not_recall():
    """Regression (opus-4.8 PR-planning, 2026-06-15): a forward-looking proposal/confirmation
    turn must NOT be flagged as recall just because it cites the past as setup (a bare 'before'
    or a number + '?'). A false positive made the gate DELETE the model's real 11-minute answer
    and ship the canned REFUSAL."""
    from cmx.enforcement import classify_factual_history as c
    forward = [
        # the exact shape that misfired: "before … so let's shoot for … right?"
        "So, before it was going to be 100 LOC, so let's shoot for <=2000 LOC per PR, "
        "then we reduce this to around 13 ~ 18 PRs max, right?",
        "before we said 100 LOC, but let's shoot for 2000 now, right?",
        "we should go with 3 retries from now on, ok?",
        "let's aim for 5 PRs instead of the 20 we had earlier, sound good?",
        "going forward we'll cap it at 2000 lines, agreed?",
    ]
    for t in forward:
        assert c(t) is False, f"false-positive recall on a forward-planning turn: {t!r}"
    # the guard must NOT swallow genuine recall that merely lacks a forward cue
    assert c("what LOC cap did we agree on before?") is True
    assert c("which value did we set earlier?") is True
