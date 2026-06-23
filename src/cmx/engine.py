"""CmxEngine — the turn-lifecycle orchestrator.

Wires the layers into one loop so grounding is a property of the engine, not a
hope about the model:

    ingest → assemble(+Layer 1 inject) → [Layer 2 forced gate] → model(+tools)
           → Layer 3 citation check → Layer 4 verify → regenerate → Layer 5 refuse

The model client is injected (real Hermes aux client in production, MockModel in
tests), so the entire pipeline is deterministically testable — including a
*guessing* model that the engine must force into grounding or refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .assembly import assemble
from .config import CmxConfig
from .enforcement import (CitationResult, Verifier, check_citations,
                          classify_factual_history, unsupported_fact_tokens)
from .llm import ModelClient
from .profiles import profile_for
from .retrieval import HybridRetriever
from .store import VerbatimStore
from .tokenizer import get_tokenizer

REFUSAL = "I don't have that in my record — want me to search wider or check a file?"

# Non-destructive degrade for refusal_mode="caveat": keep the model's real answer, append a
# short warning instead of deleting it. (refusal_mode="off" ships the answer with no note.)
_CAVEAT = ("\n\n[cmx: I couldn't fully ground every specific above in the stored record — "
           "double-check any exact values before relying on them.]")

TOOL_SCHEMAS = [
    {"name": "cmx_grep", "description": "Hybrid search over verbatim history (FTS5+trigram)."},
    {"name": "cmx_expand", "description": "Return the verbatim message for an id."},
    {"name": "cmx_recall", "description": "Return the last N verbatim turns."},
    {"name": "cmx_graph_query", "description": "Multi-hop entity/relation traversal over history."},
]


@dataclass
class Response:
    text: str
    grounded: bool
    refused: bool
    attempts: int
    evidence_ids: list[int] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    profile: str = ""


class CmxEngine:
    def __init__(self, cfg: CmxConfig, store: VerbatimStore, retriever: HybridRetriever,
                 model_client: ModelClient, verifier_client: Optional[ModelClient] = None):
        self.cfg = cfg
        self.store = store
        self.retriever = retriever
        self.model_client = model_client
        self.verifier_client = verifier_client
        self.audit: list[dict] = []

    # -- ingest -----------------------------------------------------------
    def ingest(self, session_id: str, role: str, content: str, *, pinned: bool = False,
               model: str = "") -> int:
        turn = self.store.max_turn(session_id) + 1
        tc = get_tokenizer(model or "_default").count(content) if content else 0
        return self.store.add_message(session_id, turn, role, content,
                                      pinned=pinned, token_count=tc)

    # -- tools ------------------------------------------------------------
    def _serve_tool(self, session_id: str, name: str, args: dict) -> str:
        args = args or {}
        if name == "cmx_grep":
            sl = self.retriever.retrieve(str(args.get("query", "")), session_id=session_id)
            return "\n".join(f"[id={x.id}] ({x.role}): {x.content}" for x in sl) or "no matches"
        if name == "cmx_expand":
            row = self.store.get_message(int(args.get("id", 0) or 0))
            return f"[id={row['id']}] {row['content']}" if row else "not found"
        if name == "cmx_recall":
            rows = self.store.recent(session_id, int(args.get("n", 10) or 10))
            return "\n".join(f"[id={r['id']}] ({r['role']}): {r['content']}" for r in rows)
        if name == "cmx_graph_query":
            g = getattr(self.store, "graph", None)
            if g is None:
                return "graph unavailable"
            seeds = g.entities_in(str(args.get("entity", "")))
            ids = [m for m in g.multihop_messages(seeds, self.cfg.graph_hops)
                   if self.store.in_session(m, session_id)]
            rows = [self.store.get_message(i) for i in ids[:10]]
            return "\n".join(f"[id={r['id']}] ({r['role']}): {r['content']}" for r in rows if r) \
                or "no graph matches"
        return f"unknown tool: {name}"

    def _parse_text_tool_calls(self, content: str):
        """F3: parse tool calls emitted as TEXT (no native tool API) via the 5-dialect
        registry. Returns (cleaned_text, [ToolCall]) or (content, []) when disabled/none."""
        if not getattr(self.cfg, "parse_tool_dialects", False) or not content:
            return content, []
        try:
            from .tool_dialects import normalize_tool_calls
            import json as _json
            names = {t["name"] for t in TOOL_SCHEMAS}
            cleaned, raw = normalize_tool_calls(content, tool_names=names)
            calls = []
            from .llm import ToolCall
            for tc in raw:
                fn = tc.get("function", {})
                try:
                    args = _json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                calls.append(ToolCall(name=fn.get("name", ""), args=args))
            return cleaned, calls
        except Exception:
            return content, []

    def _run_model(self, messages, model, force_tool, session_id, max_rounds=None):
        msgs = list(messages)
        served: list[str] = []
        tool_choice = "required" if force_tool else None
        rounds = max_rounds if max_rounds is not None else self.cfg.max_tool_rounds
        for _ in range(rounds):
            comp = self.model_client.complete(msgs, model=model, tool_choice=tool_choice,
                                              tools=TOOL_SCHEMAS)
            tool_choice = None  # only force the first round
            tool_calls = comp.tool_calls
            content = comp.content
            # F3 fallback: if the lane has no native tool API, the model emits the call
            # as TEXT — parse it out so forced retrieval still happens.
            if not tool_calls:
                content, tool_calls = self._parse_text_tool_calls(content)
            if tool_calls:
                for call in tool_calls:
                    result = self._serve_tool(session_id, call.name, call.args)
                    served.append(result)
                    msgs.append({"role": "tool", "content": result})
                continue
            from .forcing import strip_think_tags
            content = strip_think_tags(content, getattr(self.cfg, "strip_reasoning", False))
            return content, "\n".join(served)
        comp = self.model_client.complete(msgs, model=model)  # final, unforced
        content, _ = self._parse_text_tool_calls(comp.content)
        from .forcing import strip_think_tags
        content = strip_think_tags(content, getattr(self.cfg, "strip_reasoning", False))
        return content, "\n".join(served)

    def _self_consistent(self, messages, model, session_id, answer) -> bool:
        """L5: resample N times at temperature>0; require the primary answer's salient fact
        tokens to recur in a majority of samples. Answers with no fact token (free text) are
        not gated. Resample failure/outage → treat as consistent (don't over-refuse)."""
        from .enforcement import salient_fact_tokens, strip_citations
        primary = set(t.lower() for t in salient_fact_tokens(strip_citations(answer)))
        if not primary:
            return True  # nothing checkable
        n = self.cfg.self_consistency
        agree = 0
        for _ in range(n):
            try:
                comp = self.model_client.complete(list(messages), model=model,
                                                  temperature=self.cfg.self_consistency_temp)
                alt = set(t.lower() for t in salient_fact_tokens(strip_citations(comp.content or "")))
            except Exception:
                return True  # outage → don't block
            if primary & alt:
                agree += 1
        return agree >= (n + 1) // 2  # majority of resamples corroborate the fact

    def _evidence_text(self, ids, extra: str = "") -> str:
        parts = [(self.store.get_message(i) or {}).get("content", "") for i in ids]
        if extra:
            parts.append(extra)
        return "\n".join(p for p in parts if p)

    def _semantic_rescue(self, cres: CitationResult, verifier: Verifier) -> CitationResult:
        """Lever 6: drop citation_mismatch failures the strict NLI judge confirms are
        semantically supported (paraphrase). Phantom citations are NEVER rescued, and a
        missing/erroring judge keeps the failure (safe default)."""
        kept = []
        for f in cres.failures:
            if f.reason != "citation_mismatch":
                kept.append(f)  # phantom_citation → never soften
                continue
            row = self.store.get_message(f.message_id)
            content = (row or {}).get("content", "")
            if content and verifier.semantic_supported(
                    f.sentence, content, self.cfg.semantic_prefilter_sim):
                continue  # paraphrase confirmed by the independent judge → rescue
            kept.append(f)
        return CitationResult(ok=not kept, failures=kept)

    # -- the lifecycle ----------------------------------------------------
    def respond(self, session_id: str, user_text: str, model: str,
                system_prompt: str = "", persist: bool = True) -> Response:
        if persist:
            self.ingest(session_id, "user", user_text, model=model)
        is_factual = (bool(getattr(self.cfg, "assume_factual_history", False))
                      or classify_factual_history(user_text))
        profile = profile_for(model, self.cfg)
        tk = get_tokenizer(model)
        from .council_judge import make_verifier
        verifier = make_verifier(
            self.cfg,
            self.verifier_client if profile.verifier != "off" else None,
            self.cfg.verifier_model,
            embedder=getattr(self.retriever, "embedder", None))

        corrective = ""
        failures: list[str] = []
        last_answer = ""

        for attempt in range(self.cfg.max_regenerations + 1):
            ac = assemble(store=self.store, retriever=self.retriever, tokenizer=tk, cfg=self.cfg,
                          profile=profile, model=model, session_id=session_id,
                          user_text=user_text, system_prompt=system_prompt,
                          tool_schemas=TOOL_SCHEMAS)
            messages = list(ac.messages)

            # L2 (iter3): evidence-sufficiency PRE-gate — before spending a generation, an
            # independent judge decides whether the assembled grounding corpus actually
            # contains an answer to this question. If not, refuse straight away (no
            # confabulation opportunity). Runs once (attempt 0); deterministic retrieval ⇒
            # the corpus is the same across attempts.
            if (attempt == 0 and is_factual and self.cfg.sufficiency_gate
                    and profile.verifier != "off" and profile.refuse_to_guess
                    and getattr(self.cfg, "refusal_mode", "replace") == "replace"):
                corpus = "\n".join(str(m.get("content") or "") for m in ac.messages)
                if self.cfg.sufficiency_threshold > 0:
                    insufficient = (verifier.answerable_score(user_text, corpus)
                                    < self.cfg.sufficiency_threshold)  # L6 graded
                else:
                    insufficient = not verifier.answerable(
                        user_text, corpus, votes=self.cfg.sufficiency_votes,
                        reasoning=self.cfg.sufficiency_reasoning,
                        consensus=self.cfg.sufficiency_consensus,
                        multihop=getattr(self.cfg, "multihop", False))  # L2 / hardened
                if insufficient:
                    failures.append("evidence insufficient to answer (sufficiency gate)")
                    if persist:
                        self.ingest(session_id, "assistant", REFUSAL, model=model)
                    self._audit(session_id, REFUSAL, False, True, [])
                    return Response(REFUSAL, False, True, attempt + 1, [], failures, profile.name)

            # always present the current user turn as the final message (whether or
            # not it was persisted into the recent window)
            if not (messages and messages[-1].get("role") == "user"
                    and messages[-1].get("content") == user_text):
                messages.append({"role": "user", "content": user_text})
            if corrective:
                messages.append({"role": "system", "content": corrective})

            force = is_factual and profile.forced_gate and not ac.evidence_ids
            rounds = None
            # F5: forced+verified agentic search. On factual turns, force the model to
            # search its memory (reworded queries) before answering, with more rounds —
            # even when some evidence was injected, so a thin first injection can be
            # recovered. Kept safe by the citation check + verifier + refuse below.
            if getattr(self.cfg, "agentic_search", False) and is_factual and profile.forced_gate:
                force = True
                rounds = max(self.cfg.max_tool_rounds, getattr(self.cfg, "agentic_max_rounds", 3))
            answer, served = self._run_model(messages, model, force, session_id, max_rounds=rounds)
            last_answer = answer
            # Grounding corpus = the ENTIRE assembled verbatim context (injected
            # evidence AND the recent verbatim window) plus anything the model
            # retrieved this turn. A claim grounded in any verbatim the model could
            # see is supported — not only the injected evidence block.
            assembled_text = "\n".join(str(m.get("content") or "") for m in ac.messages)
            evidence_text = (assembled_text + "\n" + served).strip()

            if is_factual and self.cfg.strict_value_grounding:
                missing = unsupported_fact_tokens(answer, evidence_text)
                if missing:
                    corrective = ("GROUNDING ERROR: answer contains value(s) not present in "
                                  "retrieved evidence: " + ", ".join(missing[:6]) +
                                  ". Re-answer only with values explicitly present in evidence, "
                                  "or say you do not know.")
                    failures.append("strict-value grounding: " + ", ".join(missing[:6]))
                    continue

            if is_factual and self.cfg.require_citations and profile.require_citations:
                cres = check_citations(answer, self.store)
                if not cres.ok and self.cfg.semantic_verifier:
                    cres = self._semantic_rescue(cres, verifier)
                if not cres.ok:
                    corrective = ("GROUNDING ERROR: " + cres.message() +
                                  " Re-answer citing only evidence that supports each fact, "
                                  "or say you do not know.")
                    failures.append(cres.message())
                    continue

            if is_factual and profile.verifier != "off":
                vres = verifier.verify(answer, evidence_text)
                if not vres.ok:
                    corrective = ("GROUNDING ERROR: " + vres.message() +
                                  " Remove or correct unsupported claims and cite evidence, "
                                  "or say you do not know.")
                    failures.append(vres.message())
                    continue

            # L5 (iter3): self-consistency — the answer passed the deterministic checks, but
            # for a factual claim, resample and require the salient fact to be stable. An
            # on-topic-but-unanswerable confabulation drifts across samples → refuse.
            if (is_factual and self.cfg.self_consistency > 0 and profile.refuse_to_guess
                    and not self._self_consistent(messages, model, session_id, answer)):
                failures.append("self-consistency: salient fact unstable across samples")
                return self._refusal_response(session_id, model, last_answer, attempt + 1,
                                              failures, profile, persist)

            if persist:
                self.ingest(session_id, "assistant", answer, model=model)
            self._audit(session_id, answer, True, False, ac.evidence_ids)
            return Response(answer, True, False, attempt + 1, ac.evidence_ids, failures, profile.name)

        # exhausted retries → Layer 5
        if profile.refuse_to_guess:
            return self._refusal_response(session_id, model, last_answer,
                                          self.cfg.max_regenerations + 1, failures,
                                          profile, persist)
        if persist:
            self.ingest(session_id, "assistant", last_answer, model=model)
        self._audit(session_id, last_answer, False, False, [])
        return Response(last_answer, False, False, self.cfg.max_regenerations + 1, [], failures,
                        profile.name)

    def _refusal_response(self, session_id, model, last_answer, attempts,
                          failures, profile, persist):
        """Terminal grounding-failure response. refusal_mode='replace' (default) ships the
        canned REFUSAL — strict/benchmark behavior, unchanged. 'caveat' keeps the model's real
        answer and appends a short warning; 'off' ships the answer as-is. caveat/off prevent a
        misfiring gate from deleting a substantive agentic/coding answer; they fall back to
        REFUSAL when there is no answer to preserve (e.g. a pre-generation gate)."""
        mode = getattr(self.cfg, "refusal_mode", "replace")
        if mode == "caveat" and last_answer:
            text, grounded, refused = last_answer + _CAVEAT, False, False
        elif mode == "off" and last_answer:
            text, grounded, refused = last_answer, False, False
        else:
            text, grounded, refused = REFUSAL, False, True
        if persist:
            self.ingest(session_id, "assistant", text, model=model)
        self._audit(session_id, text, grounded, refused, [])
        return Response(text, grounded, refused, attempts, [], failures, profile.name)

    def _audit(self, session_id, answer, grounded, refused, evidence_ids):
        self.audit.append({"session_id": session_id, "grounded": grounded,
                           "refused": refused, "evidence_ids": list(evidence_ids),
                           "answer": answer})
