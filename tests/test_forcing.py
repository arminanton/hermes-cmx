"""Offline tests for the forcing layer (P1–P4: F1 directives, F2 protocol, F3 dialects,
F4 reasoning strip, F5 agentic loop). Deterministic — mocks only, no provider."""
from cmx.config import CmxConfig
from cmx.engine import CmxEngine, TOOL_SCHEMAS
from cmx.forcing import (memory_directives_block, strip_think_tags, tool_protocol_block)
from cmx.llm import Completion, MockModel, ToolCall
from cmx.retrieval import HybridRetriever
from cmx.store import VerbatimStore
from cmx.tool_dialects import normalize_tool_calls


# -- F1 memory directives -----------------------------------------------------

def test_memory_directives_gated():
    assert memory_directives_block(False) == ""
    blk = memory_directives_block(True)
    assert "UNLIMITED RECALL" in blk and "FAITHFUL REPORTING" in blk
    assert "I don't have that in my record" in blk


def test_tool_protocol_gated():
    assert tool_protocol_block(TOOL_SCHEMAS, False) == ""
    blk = tool_protocol_block(TOOL_SCHEMAS, True)
    assert "cmx_grep" in blk and "tool_call" in blk and "not available" in blk


# -- F3 dialect parser --------------------------------------------------------

def test_dialect_parses_nous_xml():
    _, calls = normalize_tool_calls('<tool_call>{"name":"cmx_grep","arguments":{"query":"port"}}</tool_call>',
                                    tool_names={"cmx_grep"})
    assert calls and calls[0]["function"]["name"] == "cmx_grep"


def test_dialect_parses_python_call():
    _, calls = normalize_tool_calls('cmx_grep(query="deploy port")', tool_names={"cmx_grep"})
    assert calls and calls[0]["function"]["name"] == "cmx_grep"


# -- F4 reasoning strip -------------------------------------------------------

def test_strip_think_tags():
    assert strip_think_tags("<think>secret cot</think>The port is 8080.", True) == "The port is 8080."
    # disabled = no-op
    assert strip_think_tags("<think>x</think>ans", False) == "<think>x</think>ans"
    # unterminated trailing block
    assert strip_think_tags("answer<thinking>dangling cot to EOS", True) == "answer"
    # no tags = unchanged
    assert strip_think_tags("plain answer", True) == "plain answer"


# -- engine integration -------------------------------------------------------

def _eng(tmp_path, cfg, model):
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    return CmxEngine(cfg, s, HybridRetriever(s, cfg), model, verifier_client=None), s


def test_f1_directives_injected_into_system_message(tmp_path):
    cfg = CmxConfig(); cfg.force_memory_directives = True; cfg.use_embeddings = False; cfg.rerank = False
    captured = {}

    class Capture(MockModel):
        def complete(self, messages, **kw):
            captured["sys"] = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            return Completion(content="I don't have that in my record.")

    eng, s = _eng(tmp_path, cfg, Capture(["x"]))
    eng.ingest("s", "user", "we chatted", model="gpt-5-mini")
    eng.respond("s", "what did we decide?", model="gpt-5-mini", persist=False)
    assert "UNLIMITED RECALL" in captured["sys"]


def test_f3_text_tool_call_is_served(tmp_path):
    # model emits a tool call as TEXT (no native tool_calls); F3 parses + serves it.
    cfg = CmxConfig(); cfg.parse_tool_dialects = True; cfg.use_embeddings = False; cfg.rerank = False
    cfg.recent_window_turns = 1  # push the fact out of the window so a search is meaningful
    s = VerbatimStore(str(tmp_path / "cmx.db"), use_trigram=True)
    fid = s.add_message("s", 0, "assistant", "the deploy port is 8080")
    for i in range(3):
        s.add_message("s", i + 1, "user", "unrelated chatter")
    # first call: emit a text tool call; second: answer citing the served evidence
    script = [Completion(content='<tool_call>{"name":"cmx_grep","arguments":{"query":"deploy port"}}</tool_call>'),
              Completion(content=f"The deploy port is 8080 [id={fid}].")]
    eng = CmxEngine(cfg, s, HybridRetriever(s, cfg), MockModel(script), verifier_client=None)
    out = eng._run_model([{"role": "user", "content": "port?"}], "gpt-5-mini", False, "s")
    served = out[1]
    assert "8080" in served  # the parsed tool call retrieved the verbatim evidence


def test_f5_agentic_search_forces_gate(tmp_path):
    cfg = CmxConfig(); cfg.agentic_search = True; cfg.agentic_max_rounds = 2
    cfg.use_embeddings = False; cfg.rerank = False
    seen = {}

    class Probe(MockModel):
        def complete(self, messages, *, model, tool_choice=None, tools=None, temperature=None):
            seen.setdefault("first_tool_choice", tool_choice)
            return Completion(content="I don't have that in my record.")

    eng, s = _eng(tmp_path, cfg, Probe(["x"]))
    eng.ingest("s", "user", "the port is 8080", model="gpt-5-mini")
    eng.respond("s", "what port did we decide?", model="gpt-5-mini", persist=False)
    assert seen["first_tool_choice"] == "required"  # F5 forced the search gate
