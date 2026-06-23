"""Real-model evaluation: run the cmx grounding pipeline against live models.

Compares how real models behave under cmx's proactive injection + enforcement:
each gets the same [CMX EVIDENCE] block and the same contract; the engine's
deterministic citation check + verifier catch any ungrounded claim. Measures, per
model, whether enforcement holds the SAME grounding guarantee regardless of model
strength (the model-agnosticism claim, on real models).

Bounded by design (small corpus, capped regenerations, text-only — no tool rounds;
proactive injection supplies the evidence). Verifier is deterministic (free).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cmx.config import CmxConfig            # noqa: E402
from cmx.engine import CmxEngine            # noqa: E402
from cmx.llm import Completion              # noqa: E402
from cmx.retrieval import HybridRetriever   # noqa: E402
from cmx.store import VerbatimStore         # noqa: E402

import synthetic                             # noqa: E402
import run_eval                              # noqa: E402

CONTRACT = (
    "You are answering questions about THIS conversation's history. You are given a "
    "[CMX EVIDENCE] block of verbatim excerpts, each tagged [id=N]. Rules:\n"
    "1. Answer ONLY from the evidence or the recent turns shown.\n"
    "2. Cite the source like [id=N] for every fact you state.\n"
    "3. If the answer is NOT present, reply EXACTLY: I don't have that in my record.\n"
    "Answer in one short sentence."
)


class RealModelClient:
    def __init__(self, model: str, provider: str = "", task: str = "compression",
                 max_tokens: int = 80, timeout: float = 60.0, extra_body: dict = None):
        self.model, self.provider = model, provider
        self.task, self.max_tokens, self.timeout = task, max_tokens, timeout
        self.extra_body = extra_body or None

    def complete(self, messages, *, model=None, tool_choice=None, tools=None, temperature=None) -> Completion:
        from agent.auxiliary_client import call_llm
        kw = {"task": self.task, "model": self.model, "messages": messages,
              "max_tokens": self.max_tokens, "timeout": self.timeout}
        if self.provider:
            kw["provider"] = self.provider
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        if temperature is not None:
            kw["temperature"] = temperature
        try:
            r = call_llm(**kw)
            c = r.choices[0].message.content
            return Completion(content=c if isinstance(c, str) else (str(c) if c else ""))
        except Exception as e:
            return Completion(content=f"(model error: {type(e).__name__})")


# Proper OpenAI tool schemas for the cmx retrieval tools (lever 2).
_CMX_TOOLS = [
    {"type": "function", "function": {"name": "cmx_grep",
        "description": "Search the verbatim conversation history (semantic + lexical).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "cmx_graph_query",
        "description": "Multi-hop entity/relation lookup over the history.",
        "parameters": {"type": "object", "properties": {"entity": {"type": "string"}}, "required": ["entity"]}}},
]


class ToolModelClient(RealModelClient):
    """Lever 2: gives the model the cmx retrieval tools so it can drill on its own
    when injected evidence is thin. call_llm supports `tools` (not `tool_choice`)."""
    def complete(self, messages, *, model=None, tool_choice=None, tools=None, temperature=None) -> Completion:
        from agent.auxiliary_client import call_llm
        import json as _json
        from cmx.llm import ToolCall
        kw = {"task": self.task, "model": self.model, "messages": messages,
              "max_tokens": self.max_tokens, "timeout": self.timeout, "tools": _CMX_TOOLS}
        if self.provider:
            kw["provider"] = self.provider
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        try:
            r = call_llm(**kw)
            m = r.choices[0].message
            content = m.content if isinstance(m.content, str) else (str(m.content) if m.content else "")
            calls = []
            for tc in (getattr(m, "tool_calls", None) or []):
                try:
                    calls.append(ToolCall(name=tc.function.name,
                                          args=_json.loads(tc.function.arguments or "{}")))
                except Exception:
                    pass
            return Completion(content=content, tool_calls=calls)
        except Exception as e:
            return Completion(content=f"(model error: {type(e).__name__})")


def run_real(label: str, model: str, provider: str = "", n_facts: int = 6) -> run_eval.Metrics:
    cfg = CmxConfig()
    cfg.recent_window_turns = 2
    cfg.forced_gate = "off"          # text-only; proactive injection supplies evidence
    cfg.max_regenerations = 1        # bound cost
    cfg.models = {model: {"window": 200000, "follows_instructions": "high"}} if "mini" not in model else {}
    store = VerbatimStore(run_eval._tmpdb(), use_trigram=True, use_graph=True)
    client = RealModelClient(model, provider=provider)
    eng = CmxEngine(cfg, store, HybridRetriever(store, cfg), client, verifier_client=None)

    corpus = synthetic.make_corpus(n_facts=n_facts)
    session = "eval"
    fact_ids = {}
    for role, content in corpus.turns:
        mid = eng.ingest(session, role, content, model=model)
        for f in corpus.facts:
            if content == f.sentence:
                fact_ids[f.key] = mid

    M = run_eval.Metrics(model=label)
    from cmx.profiles import profile_for
    prof = profile_for(model, cfg)
    for q in corpus.questions:
        if q.answerable and fact_ids.get(q.key) in \
                [s.id for s in eng.retriever.retrieve(q.text, k=prof.inject_k, session_id=session)]:
            M.retrieved += 1
    for q in corpus.questions:
        resp = eng.respond(session, q.text, model=model, system_prompt=CONTRACT, persist=False)
        refused = resp.refused or run_eval._is_refusal(resp.text)
        shipped = not refused
        if shipped:
            M.shipped += 1
        if shipped and run_eval._unsupported_by_store(resp.text, store, session):
            M.hallucinations += 1
        if q.answerable:
            M.answerable += 1
            if shipped and q.expected_value in resp.text:
                M.correct += 1
                M.grounded_answers += 1
                from cmx.enforcement import check_citations
                if check_citations(resp.text, store).ok:
                    M.cited_valid += 1
        else:
            M.unanswerable += 1
            if refused:
                M.correct_refusal += 1
    return M


MODELS = [
    ("opus-4.7", "claude-opus-4.7", ""),
    ("sonnet-4.5", "claude-sonnet-4.5", ""),
    ("gpt-5.5", "gpt-5.5", "copilot"),
    ("gpt-5-mini", "gpt-5-mini", ""),
]


def main():  # pragma: no cover
    out = ["# cmx REAL-model eval — synthetic planted-fact suite",
           "_Same evidence + contract + enforcement for every model._\n"]
    for label, model, provider in MODELS:
        t0 = time.time()
        try:
            m = run_real(label, model, provider)
            out.append(m.report() + f"  _(elapsed {time.time()-t0:.0f}s)_\n")
        except Exception as e:
            out.append(f"### {label}\n- ERROR: {type(e).__name__}: {e}\n")
        print(out[-1])
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(exist_ok=True)
    (d / "real-models.md").write_text("\n".join(out))
    print("wrote benchmarks/results/real-models.md")


if __name__ == "__main__":  # pragma: no cover
    main()
