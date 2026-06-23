"""cmx evaluation runner.

Demonstrates the doc-04 metrics on the synthetic corpus with two mock models:
  • GroundedModel  — reads injected evidence and cites it (cooperative).
  • GuessingModel  — fabricates plausible-but-wrong values, uncited (adversarial).

The headline result the harness verifies: enforcement drives **hallucination ≈ 0
for BOTH models** — the guessing model is converted into refusals, not confident
wrong answers. That is the model-agnosticism proof from doc 04.
"""
from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cmx.config import CmxConfig            # noqa: E402
from cmx.engine import CmxEngine            # noqa: E402
from cmx.enforcement import (               # noqa: E402
    check_citations, parse_citations, salient_fact_tokens,
)
from cmx.llm import Completion              # noqa: E402
from cmx.retrieval import HybridRetriever   # noqa: E402
from cmx.store import VerbatimStore         # noqa: E402

import synthetic  # noqa: E402

_EV_LINE = re.compile(r"\[id=(\d+)\][^\n]*?the (.+?) is (.+?)\.(?=\s|$)")
_FACT_LINE = re.compile(r"the (.+?) is (.+?)\.(?=\s|$)")
_WRONG = ["zz-bad-1", "999 seconds", "0000", "v0.0.0", "garbage-host", "wrong-q"]


def _full_text(messages):
    return "\n".join(str(m.get("content") or "") for m in messages)


def _evidence_text(messages):
    for m in messages:
        if m.get("role") == "system":
            return m.get("content") or ""
    return ""


class GroundedModel:
    """Cooperative & realistic: reads the whole prompt (injected evidence AND the
    recent verbatim window). Cites [id=N] when the fact is in the citable evidence
    block; otherwise answers from the recent window (uncited — the engine's verifier
    validates it against the in-context verbatim)."""
    def complete(self, messages, *, model, tool_choice=None, tools=None):
        q = messages[-1].get("content", "") if messages else ""
        full = _full_text(messages)
        for mid, key, val in _EV_LINE.findall(full):           # citable evidence first
            if _key_in(key, q):
                return Completion(content=f"The {key} is {val} [id={mid}].")
        for key, val in _FACT_LINE.findall(full):              # recent-window verbatim
            if _key_in(key, q):
                return Completion(content=f"The {key} is {val}.")
        return Completion(content="I don't have that in my record.")


class GuessingModel:
    """Adversarial: ignores evidence, fabricates an uncited wrong value."""
    def __init__(self, seed=1):
        self.rng = random.Random(seed)
    def complete(self, messages, *, model, tool_choice=None, tools=None):
        q = messages[-1].get("content", "") if messages else ""
        key = _extract_key(q)
        return Completion(content=f"The {key} is {self.rng.choice(_WRONG)}.")


def _extract_key(question: str) -> str:
    m = re.search(r"what (.+?) did we", question)
    return m.group(1) if m else "value"


def _key_in(key: str, question: str) -> bool:
    return all(w in question.lower() for w in key.lower().split())


@dataclass
class Metrics:
    model: str
    answerable: int = 0
    unanswerable: int = 0
    retrieved: int = 0          # recall@k numerator
    correct: int = 0            # answered correctly
    correct_refusal: int = 0
    hallucinations: int = 0
    shipped: int = 0
    cited_valid: int = 0
    grounded_answers: int = 0

    def report(self) -> str:
        rec = _pct(self.retrieved, self.answerable)
        acc = _pct(self.correct, self.answerable)
        ref = _pct(self.correct_refusal, self.unanswerable)
        hall = _pct(self.hallucinations, max(1, self.shipped))
        cite = _pct(self.cited_valid, max(1, self.grounded_answers))
        return (f"### {self.model}\n"
                f"- recall@k:            {rec:5.1f}%  ({self.retrieved}/{self.answerable})\n"
                f"- answer accuracy:     {acc:5.1f}%  ({self.correct}/{self.answerable})\n"
                f"- refusal-to-guess:    {ref:5.1f}%  ({self.correct_refusal}/{self.unanswerable})\n"
                f"- hallucination rate:  {hall:5.1f}%  ({self.hallucinations}/{self.shipped} shipped)\n"
                f"- citation validity:   {cite:5.1f}%  ({self.cited_valid}/{self.grounded_answers})\n")


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def _is_refusal(text: str) -> bool:
    t = text.lower()
    return ("don't have" in t or "do not have" in t or "not in my record" in t
            or "want me to search" in t)


def _unsupported_by_store(text: str, store, session: str) -> bool:
    """True iff the answer asserts a fact token that appears NOWHERE in the session's
    verbatim store — i.e. it was fabricated. Citation-agnostic: an uncited answer
    grounded in the recent verbatim window is NOT a hallucination."""
    for t in salient_fact_tokens(text):
        hits = (store.search_trgm(t, k=2, session_id=session)
                or store.search_fts(t, k=2, session_id=session))
        present = any(t.lower() in (store.get_message(h["id"]) or {}).get("content", "").lower()
                      for h in hits)
        if not present:
            return True
    return False


def run(model_name: str, model_client, *, n_facts=10, db_path=":memory:") -> Metrics:
    cfg = CmxConfig()
    cfg.recent_window_turns = 2   # push facts out of the window → recall measures retrieval
    cfg.models = {model_name: {"window": 200000, "follows_instructions": "high"}} \
        if "mini" not in model_name else {}
    store = VerbatimStore(db_path if db_path != ":memory:" else _tmpdb(), use_trigram=True)
    eng = CmxEngine(cfg, store, HybridRetriever(store, cfg), model_client, verifier_client=None)

    corpus = synthetic.make_corpus(n_facts=n_facts)
    session = "eval"
    fact_ids = {}
    for role, content in corpus.turns:
        mid = eng.ingest(session, role, content, model=model_name)
        for f in corpus.facts:
            if content == f.sentence:
                fact_ids[f.key] = mid

    M = Metrics(model=model_name)
    from cmx.profiles import profile_for
    prof = profile_for(model_name, cfg)

    # recall@k measured directly from the retriever (model-independent)
    for q in corpus.questions:
        if not q.answerable:
            continue
        ids = [s.id for s in eng.retriever.retrieve(q.text, k=prof.inject_k, session_id=session)]
        if fact_ids.get(q.key) in ids:
            M.retrieved += 1

    for q in corpus.questions:
        resp = eng.respond(session, q.text, model=model_name, persist=False)
        refused = resp.refused or _is_refusal(resp.text)
        shipped = not refused
        shipped_unsupported = bool(shipped and _unsupported_by_store(resp.text, store, session))
        if shipped:
            M.shipped += 1
        if shipped_unsupported:
            M.hallucinations += 1
        if q.answerable:
            M.answerable += 1
            if shipped and q.expected_value in resp.text:
                M.correct += 1
                M.grounded_answers += 1
                if check_citations(resp.text, store).ok:
                    M.cited_valid += 1
        else:
            M.unanswerable += 1
            if refused:
                M.correct_refusal += 1
    return M


def _tmpdb():
    import tempfile
    return tempfile.mktemp(suffix=".db")


def main():  # pragma: no cover
    results = [run("opus-strong", GroundedModel()), run("gpt-5-mini", GuessingModel())]
    out = ["# cmx eval — synthetic planted-fact suite\n"]
    out += [m.report() for m in results]
    text = "\n".join(out)
    print(text)
    d = Path(__file__).resolve().parent / "results"
    d.mkdir(exist_ok=True)
    (d / "latest.md").write_text(text)


if __name__ == "__main__":  # pragma: no cover
    main()
