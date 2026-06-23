"""Hermes integration: CmxContextEngine implements the host ContextEngine ABC.

What this shim does **today** (no host changes required):
  • persists every turn verbatim into the cmx store (FTS5 + trigram),
  • on compaction, returns an assembled context = pinned sinks + retrieved
    evidence (Layer-1 injection) + recent verbatim window — instead of lossy
    summaries,
  • exposes cmx_grep / cmx_expand / cmx_recall tools.

What still needs host hook **H2** (`pre_send` + post-response enforcement) for the
*full* contract: Layers 3–5 (deterministic citation check, independent verify,
refuse-to-guess regenerate). The fully-tested reference implementation of that
loop is :class:`cmx.engine.CmxEngine`; the host integration will call it once the
ABC exposes the enforcement hook. Until then this shim delivers lossless storage,
no-lossy-summary injection, and retrieval tools — already a strict improvement
over summary-replaces-verbatim.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .assembly import assemble
from .config import CmxConfig
from .profiles import profile_for
from .retrieval import HybridRetriever
from .store import VerbatimStore
from .tokenizer import get_tokenizer, window_for

try:  # import-safe without Hermes
    from agent.context_engine import ContextEngine as _Base  # type: ignore
    _HAVE_HOST = True
except Exception:  # pragma: no cover
    _Base = object  # type: ignore
    _HAVE_HOST = False

TOOL_SCHEMAS = [
    {"name": "cmx_grep", "description": "Search verbatim history (FTS5+trigram); returns cited slices.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "cmx_expand", "description": "Return the verbatim message for an id.",
     "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}},
    {"name": "cmx_recall", "description": "Return the last N verbatim turns.",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}}},
]


class CmxContextEngine(_Base):
    threshold_percent = 0.75
    protect_first_n = 3
    protect_last_n = 6

    def __init__(self, config: Optional[CmxConfig] = None, hermes_home: str = ""):
        self.cfg = config or CmxConfig.load()
        from .store_factory import make_store
        self.store = make_store(self.cfg)
        # iter2 KEEP levers, wired here and each degradable to lexical/no-op if the aux
        # client is unavailable (so a missing embedder never breaks live compaction):
        #   lever 1 embeddings + lever 4 rerank (shared embedder), lever 3 HyDE (opt-in).
        self.embedder = self._build_embedder()
        self.retriever = HybridRetriever(self.store, self.cfg, embedder=self.embedder,
                                         expander=self._build_expander())
        self.session_id = "default"
        self.model = "_default"
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self._resolved_windows: Dict[str, int] = {}
        self.context_length = window_for(self.model, self.cfg.models)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.compression_count = 0
        self._ingested = 0
        # Stage 1: auto-load durable cross-session memory into the fixed namespace
        # (idempotent content-hash dedup; a strict no-op when durable_memory is off).
        try:
            from .durable import ensure_loaded as _ensure_durable
            _ensure_durable(self.store, self.cfg)
        except Exception:
            pass

    def _build_embedder(self):
        if not (self.cfg.use_embeddings or self.cfg.rerank):
            return None
        try:
            from .embeddings import HermesEmbedder
            return HermesEmbedder(model=self.cfg.embedding_model or "text-embedding-3-small")
        except Exception:
            return None  # degrade to lexical FTS5+trigram

    def _build_expander(self):
        if not self.cfg.use_hyde:
            return None
        try:
            from .expand import ChainExpander, HydeExpander, keyword_expander
            from .llm import HermesAuxClient
            return ChainExpander(keyword_expander,
                                 HydeExpander(HermesAuxClient(task="compression"),
                                              model=self.cfg.hyde_model))
        except Exception:
            return None

    # -- identity / token state ------------------------------------------
    @property
    def name(self) -> str:
        return "cmx"

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        toks = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return bool(toks and self.threshold_tokens and toks >= self.threshold_tokens)

    # -- lifecycle --------------------------------------------------------
    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Begin/continue a session. Hermes rotates session_id on compaction and passes
        ``boundary_reason="compression"`` + ``old_session_id`` so engines can preserve
        lineage (hermes-lcm#68). We normalize to the lineage ROOT so the one logical
        conversation keeps a single effective session_id across any number of rollovers —
        otherwise retrieval (all session-scoped) would lose the prior conversation the
        instant it compacts."""
        incoming = session_id or "default"
        boundary = kwargs.get("boundary_reason")
        old_sid = kwargs.get("old_session_id")
        if boundary == "compression" and old_sid:
            # rollover: link the new id to the old id's root, store/retrieve under the root
            self.session_id = self.store.link_session(incoming, old_sid)
        else:
            # fresh start OR resume: resolve to root (a resumed child finds its history)
            self.session_id = self.store.root_session(incoming)
        self.model = kwargs.get("model") or self.model
        # Prefer a Hermes-resolved window (the authoritative live probe result)
        # for this model over the static DEFAULT_WINDOWS table. The table
        # under-reports provider-specific limits — Copilot serves Claude opus at
        # the full 1M window, not the vendor-base 200k — and on_session_start
        # fires on every start AND every compaction rollover, so re-deriving from
        # the table here used to silently clobber the 1M that update_model had
        # set. That was the root cause of the recurring "0/200k" snap-back.
        self.context_length = (
            self._resolved_windows.get(self.model)
            or window_for(self.model, self.cfg.models)
        )
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self._ingested = self.store.count(self.session_id)

    def update_model(self, model: str, context_length: int, base_url: str = "",
                     api_key: str = "", provider: str = "", api_mode: str = "") -> None:
        """Override: also refresh self.model so the tokenizer/window track a model switch
        (the ABC default updates context_length + threshold only, leaving self.model stale)."""
        self.model = model or self.model
        # Record Hermes' live-resolved window per model so on_session_start (which
        # fires on every rollover) preserves it instead of snapping back to the
        # static DEFAULT_WINDOWS table.
        if context_length:
            self.context_length = context_length
            self._resolved_windows[self.model] = context_length
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Real session boundary (CLI exit, /reset, gateway expiry). Flush any
        not-yet-stored turns so a short conversation that never hit the compaction
        threshold is still captured verbatim (content-hash dedup makes this safe)."""
        try:
            if messages:
                self._ingest_new(messages)
        except Exception:
            pass

    def on_session_reset(self) -> None:
        # cmx never purges verbatim; nothing to delete. Reset counters only.
        self.compression_count = 0
        self.last_prompt_tokens = self.last_total_tokens = self.last_completion_tokens = 0

    # -- core: ingest + Layer-1 injection assembly -----------------------
    def _ingest_new(self, messages: List[Dict[str, Any]]) -> str:
        """Idempotently ingest any non-system message not already stored for this
        (root) session. Dedup is by content hash, NOT a positional cursor: the host
        replaces the message list across a compaction (it hands back the compacted set
        cmx itself produced), so a positional cursor would miss new turns or duplicate
        old ones. Content-hash dedup is rollover-safe and idempotent. (Trade-off: two
        byte-identical short turns — e.g. "yes" twice — collapse to one stored row;
        acceptable for a retrieval store whose job is recalling facts, not turn counts.)"""
        last_user = ""
        seen = self.store.content_hashes(self.session_id)
        tk = get_tokenizer(self.model)
        for m in messages:
            role = m.get("role") or "user"
            content = m.get("content")
            if not isinstance(content, str) or not content:
                continue
            if role == "user":
                last_user = content
            if role == "system":
                continue
            h = self.store.hash_content(content)
            if h in seen:
                continue
            turn = self.store.max_turn(self.session_id) + 1
            self.store.add_message(self.session_id, turn, role, content,
                                   token_count=tk.count(content))
            seen.add(h)
        self._ingested = self.store.count(self.session_id)
        return last_user

    def compress(self, messages: List[Dict[str, Any]], current_tokens: int = None,
                 focus_topic: str = None, **kwargs) -> List[Dict[str, Any]]:
        # **kwargs absorbs host-passed args cmx doesn't use (e.g. conversation_compression.py
        # calls compress(force=...)); previously that raised TypeError and the host fell back.
        self.compression_count += 1
        system_prompt = ""
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content") or ""
        query = self._ingest_new(messages)
        profile = profile_for(self.model, self.cfg)
        tk = get_tokenizer(self.model)
        durable = self.cfg.durable_session if getattr(self.cfg, "durable_memory", False) else None
        ac = assemble(store=self.store, retriever=self.retriever, tokenizer=tk, cfg=self.cfg,
                      profile=profile, model=self.model, session_id=self.session_id,
                      user_text=query or (focus_topic or ""), system_prompt=system_prompt,
                      durable_session=durable)
        return ac.messages

    def durable_context(self, user_message: str) -> str:
        """Render the durable cross-session memory relevant to ``user_message`` as an
        injectable block — for EVERY-TURN injection via the plugin's pre_llm_call hook.

        Why this exists: durable memory (MEMORY.md/USER.md) replaced the always-present
        ~200K system-prompt dump. But the engine only assembles/injects inside compress()
        — the COMPACTION hook — which no longer fires now the prompt is small. So without
        this, durable facts reach the model only on the rare compaction turn (the
        "what TTS voice did I pick?" deflection). The plugin calls this each turn so the
        relevant durable slices are always present. Namespaced + bm25-ranked, so transient
        tool-output never competes with it; '' when disabled / nothing relevant."""
        if not getattr(self.cfg, "durable_memory", False):
            return ""
        if not user_message or not user_message.strip():
            return ""
        try:
            from .assembly import _render_durable
            slices = self.retriever.retrieve(
                user_message, k=int(getattr(self.cfg, "durable_inject_k", 6) or 6),
                session_id=self.cfg.durable_session)
            return _render_durable(slices) if slices else ""
        except Exception:
            return ""

    def conversation_context(self, user_message: str,
                             messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """Render the relevant slice of THIS session's stored conversation as an injectable
        block — for EVERY-TURN injection via the plugin's pre_llm_call hook.

        Why this exists: cmx's conversation retrieval ran ONLY inside compress() (the
        compaction hook, which fires at 75% of the window = 300K tokens for a 400K model), so
        a normal multi-turn chat never triggered it. Worse, providers that truncate raw
        chat_history server-side (some hosted proxies keep only ~20 entries) silently drop
        early turns. So at
        25+ turns the model 'forgot' facts from turns 1-5 even though they were in the store.
        This retrieves + injects the relevant stored turns EVERY turn regardless of token
        count, so mid-conversation memory survives provider-side truncation. It also INGESTS
        the current messages first, so the store is populated even on chats that never compact
        or call a tool (compress()/handle_tool_call were the only other ingest paths).

        Not deduped against the visible window on purpose: the host's message list is NOT what
        the model sees after a truncating proxy trims it server-side, so re-injecting a
        'visible' early turn is exactly the point. '' when disabled / nothing relevant."""
        if not getattr(self.cfg, "inject_conversation", True):
            return ""
        if not user_message or not user_message.strip():
            return ""
        try:
            if messages:
                self._ingest_new(messages)  # capture history even if compress() never fires
            from .assembly import _render_evidence
            from .retrieval import Slice
            k = int(getattr(self.cfg, "conversation_inject_k", 6) or 6)
            query_slices = self.retriever.retrieve(
                user_message, k=k, session_id=self.session_id)

            # Recency floor: ALWAYS include the last N stored turns, deduped against
            # the query hits. Without this, a low-signal continuation ("ok", "go on",
            # "what's next") retrieves scattered lexical matches from anywhere in the
            # history and the actual recent context is never injected — the model
            # 'forgets' what it was just doing even though the turns are stored.
            floor_n = int(getattr(self.cfg, "conversation_recent_floor", 8) or 0)
            recent_slices: list = []
            if floor_n > 0:
                seen = {s.id for s in query_slices}
                for r in self.store.recent(self.session_id, floor_n):
                    if r["id"] in seen:
                        continue
                    recent_slices.append(Slice(
                        id=r["id"], content=r["content"], score=0.0,
                        role=r["role"], turn_index=r["turn_index"],
                        sources=("recent",)))

            # Order: query-relevant evidence first (most semantically pertinent),
            # then the recent tail in chronological order so the model reads the
            # immediate context last (closest to its own next turn).
            recent_slices.sort(key=lambda s: s.turn_index)
            combined = query_slices + recent_slices
            return _render_evidence(combined) if combined else ""
        except Exception:
            return ""

    # -- tools ------------------------------------------------------------
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(TOOL_SCHEMAS)

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        msgs = kwargs.get("messages")
        if msgs:
            self._ingest_new(msgs)
        args = args or {}
        if name == "cmx_grep":
            q = str(args.get("query", ""))
            sl = self.retriever.retrieve(q, session_id=self.session_id)
            # on-demand reach into durable memory too, so the FULL durable content is always
            # retrievable (nothing is ever unavailable — only un-proactively-injected).
            if getattr(self.cfg, "durable_memory", False):
                have = {s.id for s in sl}
                sl = sl + [s for s in self.retriever.retrieve(q, session_id=self.cfg.durable_session)
                           if s.id not in have]
            return json.dumps({"matches": [s.as_evidence() for s in sl]})
        if name == "cmx_expand":
            row = self.store.get_message(int(args.get("id", 0) or 0))
            return json.dumps(row or {"error": "not found"})
        if name == "cmx_recall":
            rows = self.store.recent(self.session_id, int(args.get("n", 10) or 10))
            return json.dumps({"turns": rows})
        return json.dumps({"error": f"unknown tool {name}"})

    def get_status(self) -> Dict[str, Any]:
        return {"engine": "cmx", "last_prompt_tokens": self.last_prompt_tokens,
                "threshold_tokens": self.threshold_tokens, "context_length": self.context_length,
                "compression_count": self.compression_count,
                "verbatim_rows": self.store.count(self.session_id),
                "trigram": self.store.has_trigram}

    # -- H2 grounding hooks ----------------------------------------------
    def capabilities(self) -> Dict[str, bool]:
        # cmx injects in compress(); the high-value hook is post-response enforcement.
        # Gated by refusal_mode: 'off' WITHDRAWS the capability so the host never calls
        # enforce_response at all — zero overhead, zero answer-replacement (the master
        # off-switch for agentic/coding use). 'replace'/'caveat' keep it advertised.
        return {"enforce_response": getattr(self.cfg, "refusal_mode", "replace") != "off"}

    def enforce_response(self, answer: str, messages: List[Dict[str, Any]],
                         model: str = "", **kwargs) -> Dict[str, Any]:
        """Layers 3-5 in-host: deterministic citation check + verification against
        the assembled verbatim context, returning a regenerate/refuse directive the
        host loop acts on. Non-factual or empty answers pass through."""
        from .enforcement import (CitationResult, Verifier, check_citations,
                                  classify_factual_history, unsupported_fact_tokens)
        from .engine import REFUSAL
        if not answer or not answer.strip():
            return {"action": "accept"}
        # only enforce when the latest user turn was a history question
        last_user = next((m.get("content", "") for m in reversed(messages)
                          if m.get("role") == "user"), "")
        if not classify_factual_history(last_user):
            return {"action": "accept"}
        final = bool(kwargs.get("final"))
        full = "\n".join(str(m.get("content") or "") for m in messages)

        # Ground the verifier against what cmx ACTUALLY KNOWS, not just the passed messages.
        # The host may inject durable memory / retrieved evidence into the user turn upstream
        # (pre_llm_call), so it isn't always in `messages`. Without this, a question the model
        # answered correctly from injected durable/session memory ("what TTS voice did I pick?")
        # would be refused because the verifier's corpus didn't contain the evidence. We augment
        # the corpus with cmx's own retrieval over the session AND the durable namespace.
        try:
            ev = self.retriever.retrieve(last_user, session_id=self.session_id)
            if getattr(self.cfg, "durable_memory", False):
                ev = ev + self.retriever.retrieve(last_user, session_id=self.cfg.durable_session)
            if ev:
                full = full + "\n" + "\n".join(s.content for s in ev)
        except Exception:
            pass

        # lever 6 (opt-in): an independent judge enables the semantic rescue (paraphrase)
        # AND audits uncited claims. Default-off → deterministic-only (judge=None) as before.
        # IMPORTANT: the judge should be a CAPABLE model — a weak model judging a weak
        # model is the residual-hallucination weak link (iter2 finding). cfg.verifier_model.
        judge = None
        if self.cfg.semantic_verifier or self.cfg.sufficiency_gate:
            try:
                from .llm import HermesAuxClient
                judge = HermesAuxClient(task="compression")
            except Exception:
                judge = None
        verifier = Verifier(judge, self.cfg.verifier_model, embedder=self.embedder)

        # iter3 L2 (opt-in): evidence-sufficiency check. The deterministic checks confirm
        # "the claim is supported by the cited slice" but NOT "this evidence actually answers
        # the question" — so an on-topic-but-unanswerable confabulation can pass them. A
        # capable judge decides sufficiency; if the evidence does not answer the question,
        # refuse (post-hoc, via this existing hook — no new host hook needed). Needs a
        # CAPABLE judge (cfg.verifier_model, e.g. gpt-5.4).
        if self.cfg.sufficiency_gate and judge is not None:
            try:
                if not verifier.answerable(last_user, full, votes=self.cfg.sufficiency_votes,
                                           reasoning=self.cfg.sufficiency_reasoning,
                                           consensus=self.cfg.sufficiency_consensus):
                    if final:
                        return self._refusal_verdict(answer)
                    return {"action": "regenerate",
                            "message": "GROUNDING ERROR: the retrieved evidence does not "
                                       "actually answer the question. If it is not in your "
                                       "record, say so — do not guess."}
            except Exception:
                pass

        if self.cfg.strict_value_grounding:
            missing = unsupported_fact_tokens(answer, full)
            if missing:
                if final:
                    return self._refusal_verdict(answer)
                return {"action": "regenerate",
                        "message": "GROUNDING ERROR: answer contains value(s) not present "
                                   "in evidence: " + ", ".join(missing[:6]) +
                                   ". Re-answer only with values explicitly present in "
                                   "evidence, or say you do not know."}

        cres = check_citations(answer, self.store)
        if not cres.ok and self.cfg.semantic_verifier and judge is not None:
            kept = []
            for f in cres.failures:
                if f.reason != "citation_mismatch":
                    kept.append(f)  # phantom citations are NEVER softened
                    continue
                row = self.store.get_message(f.message_id)
                content = (row or {}).get("content", "")
                if content and verifier.semantic_supported(
                        f.sentence, content, self.cfg.semantic_prefilter_sim):
                    continue
                kept.append(f)
            cres = CitationResult(ok=not kept, failures=kept)
        if not cres.ok:
            if final:
                return self._refusal_verdict(answer)
            return {"action": "regenerate",
                    "message": "GROUNDING ERROR: " + cres.message() +
                               " Re-answer citing only evidence that supports each fact, "
                               "or say you do not know."}
        vres = verifier.verify(answer, full)
        if not vres.ok:
            if final:
                return self._refusal_verdict(answer)
            return {"action": "regenerate",
                    "message": "GROUNDING ERROR: " + vres.message() +
                               " Remove or correct unsupported claims and cite evidence, "
                               "or say you do not know."}
        return {"action": "accept"}

    def _refusal_verdict(self, answer: str) -> Dict[str, Any]:
        """Honor cfg.refusal_mode for a grounding-failure on a FINAL answer. 'replace'
        (default, benchmark/strict) ships the canned REFUSAL; 'caveat' keeps the model's
        real answer with a short warning appended (never deletes it); 'off' accepts the
        answer untouched. Stops a misfiring grounding gate from destroying a substantive
        agentic/coding answer (the opus-4.8 11-minute-then-nuked regression, 2026-06-15)."""
        from .engine import REFUSAL, _CAVEAT
        mode = getattr(self.cfg, "refusal_mode", "replace")
        if mode == "off":
            return {"action": "accept"}
        if mode == "caveat":
            return {"action": "replace", "text": (answer or "") + _CAVEAT}
        return {"action": "replace", "text": REFUSAL}


def register(ctx=None):  # pragma: no cover - plugin entrypoint
    """Hermes plugin entrypoint. Builds and registers the cmx context engine."""
    engine = CmxContextEngine()
    if ctx is not None and hasattr(ctx, "register_context_engine"):
        ctx.register_context_engine(engine)
    return engine
