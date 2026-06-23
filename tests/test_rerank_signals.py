"""Offline tests for the temporal + importance re-ranking signals (iter4 accuracy levers)."""
from cmx.config import CmxConfig
from cmx.rerank_signals import (importance_signal, temporal_direction,
                                temporal_signal, query_years)
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore


# -- temporal signal ----------------------------------------------------------

def test_temporal_direction():
    assert temporal_direction("what did we recently decide?") == 1
    assert temporal_direction("what was our first plan?") == -1
    assert temporal_direction("what is the deploy port?") == 0
    # mixed → neutral
    assert temporal_direction("from first to most recent") == 0


def test_temporal_signal_recent_prefers_late():
    # recent intent: a late turn scores higher than an early one
    late = temporal_signal("the recent trip", turn_index=90, max_turn=100)
    early = temporal_signal("the recent trip", turn_index=10, max_turn=100)
    assert late > 0 > early


def test_temporal_signal_first_prefers_early():
    late = temporal_signal("our first apartment", turn_index=90, max_turn=100)
    early = temporal_signal("our first apartment", turn_index=10, max_turn=100)
    assert early > 0 > late


def test_temporal_signal_year_anchor():
    assert query_years("what happened in 2023?") == {"2023"}
    s = temporal_signal("what happened in 2023?", 50, 100, content="[3 May 2023] we moved house")
    assert s > 0  # year match boosts


def test_temporal_signal_no_intent_is_zero():
    assert temporal_signal("the deploy port", 50, 100, content="port 8080") == 0.0


# -- importance signal --------------------------------------------------------

def test_importance_decision_outranks_chatter():
    decision = importance_signal("We decided to deploy on port 8080 next Friday.")
    chatter = importance_signal("haha yeah the weather is nice today")
    assert decision > chatter
    assert decision > 0.4


def test_importance_bounded():
    s = importance_signal("I decided I bought booked moved started 8080 2023 prod-db v2")
    assert 0.0 <= s <= 1.0


# -- retrieval integration ----------------------------------------------------

def test_retrieve_applies_levers_without_breaking(tmp_path):
    cfg = CmxConfig()
    cfg.use_embeddings = False
    cfg.temporal_rerank = True
    cfg.importance_rerank = True
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    s.add_message("x", 0, "user", "early on we chose Postgres for the database")
    for i in range(1, 8):
        s.add_message("x", i, "user", "small talk about lunch")
    s.add_message("x", 8, "user", "recently we switched the database to MySQL")
    r = HybridRetriever(s, cfg)
    hits = r.retrieve("which database did we recently choose?", k=5, session_id="x")
    assert hits  # levers don't break retrieval; recent MySQL turn should surface
    assert any("MySQL" in h.content for h in hits)


# -- multi-hop lever (iter4 lever 2) ------------------------------------------

def test_multihop_raises_inject_k():
    from cmx.profiles import profile_for
    cfg = CmxConfig()
    base = profile_for("gpt-5-mini", cfg).inject_k
    cfg.multihop = True
    assert profile_for("gpt-5-mini", cfg).inject_k > base  # 4 -> multihop_inject_k


def test_multihop_gate_allows_combining_but_refuses_missing():
    from cmx.enforcement import Verifier
    from cmx.llm import MockModel
    # the multihop gate prompt is used; judge says YES (derivable) -> answerable
    v = Verifier(MockModel(["needs A and B\nANSWER: YES"]), "m")
    assert v.answerable("how old is X vs Y?", "X born 1980 [id=1]\nY born 1990 [id=2]",
                        reasoning=True, multihop=True) is True
    # judge says a needed fact is missing -> NO
    v2 = Verifier(MockModel(["needs Z\nANSWER: NO"]), "m")
    assert v2.answerable("q", "on-topic but missing Z", reasoning=True, multihop=True) is False


def test_multihop_instruction_injected(tmp_path):
    from cmx.engine import CmxEngine, TOOL_SCHEMAS
    from cmx.assembly import assemble
    from cmx.retrieval import HybridRetriever
    from cmx.store import VerbatimStore
    from cmx.profiles import profile_for
    from cmx.tokenizer import get_tokenizer
    cfg = CmxConfig(); cfg.multihop = True; cfg.use_embeddings = False; cfg.rerank = False
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    s.add_message("x", 0, "user", "fact one")
    ac = assemble(store=s, retriever=HybridRetriever(s, cfg), tokenizer=get_tokenizer("gpt-5-mini"),
                  cfg=cfg, profile=profile_for("gpt-5-mini", cfg), model="gpt-5-mini",
                  session_id="x", user_text="combine?", system_prompt="SYS", tool_schemas=TOOL_SCHEMAS)
    assert "MULTI-HOP" in ac.messages[0]["content"]
