"""cmx configuration.

Precedence: CMX_* env vars  >  a ``cmx:`` block in <HERMES_HOME>/config.yaml  >
defaults. This fixes the LCM gap where only env vars were honored (no YAML block).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:
    import yaml  # optional
except Exception:  # pragma: no cover
    yaml = None


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


@dataclass
class CmxConfig:
    database_path: str = ""                 # "" => <HERMES_HOME>/cmx.db (SQLite default)
    backend: str = "sqlite"                 # "sqlite" (default, zero-config) or "postgres" (scale)
    pg_dsn: str = ""                        # postgres DSN; used only when backend=postgres
                                            # (falls back to CMX_PG_DSN / SESSIONS_PG_DSN env)

    # retrieval  (iter2 ship config: embeddings + rerank ON by default — both degrade
    # gracefully to lexical FTS5+trigram if the aux embedder is unavailable; together they
    # lift evidence-recall@8 from ~55% (lexical) to ~77%. HyDE adds another ~+5-7pts but
    # costs one cheap model call per retrieval, so it is opt-in.)
    use_trigram: bool = True
    use_embeddings: bool = True             # lever 1 (KEEP): semantic kNN accelerator
    use_graph: bool = True                  # P5 entity/relation multi-hop
    graph_hops: int = 2
    embedding_model: str = ""
    use_hyde: bool = False                  # lever 3 (KEEP, opt-in): HyDE query expansion
    hyde_model: str = "gpt-5-mini"          # cheap model for the hypothetical-answer expansion
    use_chunks: bool = False                # iter3 L3: chunk-level embeddings for long turns
    chunk_min_chars: int = 600              # only chunk messages longer than this
    chunk_size_chars: int = 400             # target chunk size (chars)
    fusion: str = "rrf"
    rrf_k: int = 60                         # RRF constant
    rerank: bool = True                     # lever 4 (KEEP): embedding-cosine rerank of the pool
    rerank_weight: float = 1.0
    llm_rerank: bool = False                # iter3 L4: LLM relevance reranker (opt-in)
    llm_rerank_model: str = "gpt-5-mini"
    k_default: int = 8
    recency_weight: float = 0.05
    pin_boost: float = 0.10

    # iter4 accuracy levers (Dakera-style, deterministic; see external-benchmark-investigation.md).
    # Query-aware temporal re-ranking: re-weight candidates by how their age matches the query's
    # temporal intent ("recent" → late turns, "first/originally" → early, explicit year → date match).
    temporal_rerank: bool = False
    temporal_weight: float = 0.15
    # Importance weighting: boost load-bearing memories (decisions/commitments/facts) over chatter.
    importance_rerank: bool = False
    importance_weight: float = 0.15

    # iter4 lever 2: MULTI-HOP synthesis. The diagnostic showed cmx is near-ceiling on single-hop
    # (76%) but collapses on multi-hop (35%, mostly REFUSALS) because the iter-3 sufficiency gate
    # asks "is the answer stated in the evidence?" and refuses when the answer must be DERIVED by
    # combining 2+ slices. multihop=True reframes the gate to "derivable from the cited evidence",
    # instructs the model to combine evidence across [id=N] slices, and raises the injected-evidence
    # cap so the linked slices are both present. SAFETY held by the unchanged citation-check +
    # verifier: every sub-fact must be real and cited, so an adversarial question (a needed fact
    # absent from all slices) still fails grounding and is refused.
    multihop: bool = False
    multihop_inject_k: int = 8        # raise inject cap for strict profiles when multihop (default cap=4)

    # assembly / budgeting
    recent_window_turns: int = 12
    reserve_output_tokens: int = 4096
    safety_margin_tokens: int = 1024
    # Stage 3: room cmx leaves for host-side context it does NOT place in its assembled
    # messages — chiefly the tools[] array (capped by tool_search) and anything else added
    # after compaction. cmx counts its own messages (incl. the system prompt + skills index)
    # but not the separately-passed tools[], so without this it can under-budget on heavy-tool
    # / small-window setups. Set higher for prompted-tool models (no native tool API).
    reserve_overhead_tokens: int = 0

    # enforcement
    require_citations: bool = True
    forced_gate: str = "auto"               # auto|always|off
    verifier_model: str = "gpt-5-mini"
    verifier_mode: str = "profile"          # profile|always|off
    refuse_to_guess: bool = True
    # When a grounding gate fails on a factual turn, what to SHIP. "replace" (default,
    # benchmark/strict) substitutes the canned REFUSAL. "caveat" keeps the model's real
    # answer and appends a short non-destructive warning. "off" ships the answer as-is.
    # caveat/off stop a misfiring gate from DELETING a substantive agentic/coding answer.
    refusal_mode: str = "replace"           # replace|caveat|off
    max_regenerations: int = 2
    max_tool_rounds: int = 3

    # lever 6 (iter2): semantic-support rescue for the deterministic checks.
    # When a CITED sentence fails the verbatim fact-token check (paraphrase/synonym), an
    # independent strict NLI judge gets one chance to confirm the cited evidence actually
    # supports it — reducing FALSE refusals on correctly-grounded free-form answers. This
    # is a CHECK (default UNSUPPORTED on doubt/outage), not a delegation: phantom citations
    # are NEVER softened, and the judge is prompted to flag contradiction/substitution
    # (e.g. "chose Postgres" vs evidence "chose MySQL" -> UNSUPPORTED). Also the safety key
    # that makes the tool loop (lever 2) sound: it audits whether retrieved evidence truly
    # answers the question, not just whether a token appears.
    semantic_verifier: bool = False
    semantic_prefilter_sim: float = 0.0      # >0: skip judge if claim/evidence cosine < this (stays failed)

    # L5 (iter3): self-consistency refusal. Resample the answer N times at temperature>0;
    # if the salient fact drifts across samples, refuse — grounded answers are stable
    # (anchored by evidence), confabulations are unstable. Targets the residual where a
    # weak model invents an answer to an on-topic-but-unanswerable question.
    self_consistency: int = 0                # 0=off; N extra samples
    self_consistency_temp: float = 0.7

    # L2 (iter3): evidence-sufficiency PRE-gate. Before generating, an independent judge
    # decides whether the assembled evidence actually contains an answer to THIS question;
    # if not, refuse without calling the foreground model. Targets the residual where an
    # on-topic-but-unanswerable question retrieves topically-similar (but insufficient)
    # evidence and a weak model stitches a confabulated answer (L1 showed a similarity
    # threshold can't separate these — this is the semantic check that can).
    sufficiency_gate: bool = False
    # L6 (iter3): graded sufficiency confidence. 0 => binary YES/NO answerable() (L2);
    # >0 => the judge rates 0-100 how confidently the evidence answers the question, and we
    # refuse below this threshold — a tunable operating point on the accuracy/safety trade.
    sufficiency_threshold: int = 0
    # iter3-hardened sufficiency gate: reasoning/decomposition judge + consensus vote, to
    # cut the judge's on-topic false-positive (the residual's root cause).
    sufficiency_votes: int = 1               # K judgments; unanimous 'answerable' to proceed
    sufficiency_reasoning: bool = False      # judge names the specific fact, then checks it
    sufficiency_consensus: str = "unanimous" # unanimous|majority

    # iter4: JUDGE BACKEND. "single" = the built-in single-model Verifier (iter3 L2/L4).
    # "council" = the Hermes Council multi-judge (single-model, many-souls) wired in as the
    # sufficiency gate (L2) and the Layer-4 verifier. Targets the iter3 residual (on-topic-but-
    # unanswerable) the single judge could not eliminate. Requires hermes_council importable
    # (set COUNCIL_SRC) and the Council's own COUNCIL_PROVIDER/COUNCIL_MODEL configured (e.g.
    # COUNCIL_PROVIDER=hermes, COUNCIL_MODEL=claude-opus-4.7 → one capable model wears every
    # soul). The injected single judge becomes the fallback if the Council is unreachable.
    judge_backend: str = "single"           # single|council
    council_mode: str = "fast"              # fast|standard|deep (hot path = fast)
    council_verdict_only: bool = True       # compact, low-latency Council path
    council_panel: str = ""                 # optional curated panel (e.g. anti_sycophancy)
    council_preset: str = ""                # optional preset alias (e.g. adversarial)
    council_peer_review: bool = False       # add the anonymous Borda peer-review round
    council_personas: list = field(default_factory=list)  # explicit deliberator seats (member-count study)

    # FORCING LAYER (prompt-level grounding-enforcement blocks; see src/cmx/forcing.py).
    # Prompt-level forcing that makes a reluctant model actually USE its verbatim memory instead
    # of refusing / claiming "no memory" / claiming "tool unavailable". All flag-gated, default OFF.
    force_memory_directives: object = False  # F1: False|'full'|'lite'|'recall' (memory <system-reminder>)
    prompted_tool_protocol: bool = False    # F2: inject the M2 prompted <tools> protocol for forced retrieval
    parse_tool_dialects: bool = False       # F3: parse tool calls from model TEXT (5-dialect registry)
    strip_reasoning: bool = False           # F4: strip leaked <think>…</think> from visible output
    agentic_search: bool = False            # F5: forced+verified agentic memory-search loop
    agentic_max_rounds: int = 3             # F5: max forced search rounds before answering

    # The heuristic factual-history classifier biases toward enforcement but under-fires on
    # pure conversation-history QA benchmarks (LOCOMO/OOLONG), where EVERY question is about
    # history. Set True to treat every turn as factual-history so the sufficiency gate +
    # verifier always engage — required for a meaningful judge A/B and for memory-only
    # deployments. Default False keeps the heuristic (no behavior change).
    assume_factual_history: bool = False

    # iter3 strict value-grounding backstop: every salient fact token in the shipped answer
    # (dates, numbers, names, identifiers — the confabulation surface) MUST appear verbatim in
    # the assembled evidence corpus, else refuse. Deterministic, model-independent backstop
    # against confabulated VALUES that slip past the judges. Opt-in (can over-refuse
    # paraphrased values).
    strict_value_grounding: bool = False

    # lever 5 (iter2, REJECTED as a default): evidence count vs enforcement strictness.
    # We tested raising strict-profile inject_k (4 -> k_default) for weak-but-large-window
    # models (gpt-5-mini @128k). RESULT: NO accuracy gain and reproducibly MORE
    # confabulation (hallucination ~4.5%->~17%, adversarial-refusal ~92%->~75%) — weak
    # models synthesize false answers from the extra slices. So strict models keep a
    # conservative evidence cap regardless of window; precision (right evidence in top-k,
    # levers 1/3/4), not recall, is what helps them. strict_inject_k>0 is an opt-in
    # override (e.g. for a future well-behaved small model); default keeps the cap.
    strict_inject_k: int = 0                 # 0 => conservative cap (below); >0 overrides
    strict_inject_k_small: int = 4           # the conservative strict-profile evidence cap

    # models registry (window/strength/etc.) — overridable via YAML
    models: dict[str, Any] = field(default_factory=dict)

    # Stage 1 — durable cross-session memory (e.g. MEMORY.md/USER.md) via cmx retrieval, NOT a
    # verbatim system-prompt dump. Ingested verbatim into a fixed namespace that never rotates;
    # each turn cmx retrieves the relevant slices, and the FULL content stays reachable on demand
    # via cmx_grep. NEVER dropped or summarized — the budget only bounds how much is PROACTIVELY
    # injected; everything else is one search-tool call away, verbatim.
    durable_memory: bool = False                          # master switch for the durable namespace
    durable_session: str = "__cmx_durable__"              # fixed namespace id (never rotates)
    durable_inject_k: int = 12                            # proactive durable slices/turn (on-demand reaches the rest)
    durable_sources: list = field(default_factory=list)   # [{path, role}] docs to ingest verbatim
    durable_chunk_chars: int = 700                        # target chunk size when splitting a doc

    # Converged design — light teaching directive: proactively inject the relevant slice (above)
    # then add ONE short note teaching the model to fetch more on demand (cmx_grep / tool_search /
    # skills_list+skill_view) instead of dumping lists or forcing a protocol. Low-token,
    # low-regression; default-on. (The heavy MEMORY_DIRECTIVES/TOOL_PROTOCOL forcing stays off.)
    teach_retrieval: bool = True

    # Every-turn LIVE-conversation retrieval (parallel to durable memory). cmx's conversation
    # retrieval used to run ONLY inside compress() (fires at 75% of the window = 300K tokens for a
    # 400K model), so a normal multi-turn chat never triggered it — and providers that truncate
    # raw history server-side (some hosted proxies cap at ~20 entries) silently dropped early
    # turns. This injects
    # the relevant slice of THIS session's stored turns EVERY turn regardless of token count, so
    # mid-conversation memory survives provider truncation. This is cmx's core retrieval-first
    # promise, now wired for live conversation, not just durable memory.
    inject_conversation: bool = True
    conversation_inject_k: int = 12
    # Recency floor for the every-turn conversation injection: ALWAYS include the
    # last N stored turns of THIS session, fused with (and deduped against) the
    # query-similarity results. Fixes the low-signal-query blind spot — when the
    # current message is a vague continuation ("ok", "continue", "what's next"),
    # query-retrieval alone scatters across the whole history and never surfaces
    # the actual recent context, so the model 'forgets' what it was just doing.
    # The compaction path (assemble()) already does this via recent_window_turns;
    # this gives the per-turn hook the same recency guarantee. 0 disables.
    conversation_recent_floor: int = 8

    def resolved_db_path(self) -> str:
        return self.database_path or str(_hermes_home() / "cmx.db")

    def resolved_backend(self) -> str:
        # SQLite by default (zero-config). Postgres is opt-in: selected when
        # backend is explicitly "postgres" OR a DSN is provided via config/env.
        backend = (self.backend or os.environ.get("CMX_BACKEND", "")).strip().lower()
        if backend == "postgres" or self.resolved_pg_dsn():
            return "postgres"
        return "sqlite"

    def resolved_pg_dsn(self) -> str:
        return (self.pg_dsn
                or os.environ.get("CMX_PG_DSN", "")
                or os.environ.get("SESSIONS_PG_DSN", "")
                or os.environ.get("HERMES_SESSIONS_PG_DSN", ""))

    # -- loading ----------------------------------------------------------
    @classmethod
    def load(cls, *, env: dict | None = None, yaml_path: Path | None = None) -> "CmxConfig":
        env = os.environ if env is None else env
        cfg = cls()
        cfg._apply_yaml(yaml_path)
        cfg._apply_env(env)
        return cfg

    def _apply_yaml(self, yaml_path: Path | None) -> None:
        path = yaml_path or (_hermes_home() / "config.yaml")
        if yaml is None:
            return
        try:
            data = yaml.safe_load(Path(path).read_text()) or {}
        except Exception:
            return
        block = (data.get("cmx") or {}) if isinstance(data, dict) else {}
        names = {f.name for f in fields(self)}
        for k, v in block.items():
            if k in names:
                setattr(self, k, v)

    def _apply_env(self, env) -> None:
        def b(v: str) -> bool:
            return v.strip().lower() in {"1", "true", "yes", "on"}

        mapping = {
            "CMX_DATABASE_PATH": ("database_path", str),
            "CMX_BACKEND": ("backend", str),
            "CMX_PG_DSN": ("pg_dsn", str),
            "CMX_USE_TRIGRAM": ("use_trigram", b),
            "CMX_USE_EMBEDDINGS": ("use_embeddings", b),
            "CMX_EMBEDDING_MODEL": ("embedding_model", str),
            "CMX_K_DEFAULT": ("k_default", int),
            "CMX_RECENT_WINDOW_TURNS": ("recent_window_turns", int),
            "CMX_CONVERSATION_RECENT_FLOOR": ("conversation_recent_floor", int),
            "CMX_REQUIRE_CITATIONS": ("require_citations", b),
            "CMX_FORCED_GATE": ("forced_gate", str),
            "CMX_VERIFIER_MODEL": ("verifier_model", str),
            "CMX_VERIFIER_MODE": ("verifier_mode", str),
            "CMX_REFUSE_TO_GUESS": ("refuse_to_guess", b),
            "CMX_REFUSAL_MODE": ("refusal_mode", str),
            "CMX_MAX_REGENERATIONS": ("max_regenerations", int),
            "CMX_RESERVE_OUTPUT_TOKENS": ("reserve_output_tokens", int),
            "CMX_RERANK": ("rerank", b),
            "CMX_USE_HYDE": ("use_hyde", b),
            "CMX_HYDE_MODEL": ("hyde_model", str),
            "CMX_STRICT_INJECT_K": ("strict_inject_k", int),
            "CMX_SEMANTIC_VERIFIER": ("semantic_verifier", b),
            "CMX_SUFFICIENCY_GATE": ("sufficiency_gate", b),
            "CMX_SUFFICIENCY_VOTES": ("sufficiency_votes", int),
            "CMX_SUFFICIENCY_REASONING": ("sufficiency_reasoning", b),
            "CMX_SUFFICIENCY_CONSENSUS": ("sufficiency_consensus", str),
            "CMX_STRICT_VALUE_GROUNDING": ("strict_value_grounding", b),
            "CMX_JUDGE_BACKEND": ("judge_backend", str),
            "CMX_COUNCIL_MODE": ("council_mode", str),
            "CMX_COUNCIL_VERDICT_ONLY": ("council_verdict_only", b),
            "CMX_COUNCIL_PANEL": ("council_panel", str),
            "CMX_COUNCIL_PRESET": ("council_preset", str),
            "CMX_COUNCIL_PEER_REVIEW": ("council_peer_review", b),
            "CMX_ASSUME_FACTUAL_HISTORY": ("assume_factual_history", b),
            "CMX_FORCE_MEMORY_DIRECTIVES": ("force_memory_directives", b),
            "CMX_PROMPTED_TOOL_PROTOCOL": ("prompted_tool_protocol", b),
            "CMX_PARSE_TOOL_DIALECTS": ("parse_tool_dialects", b),
            "CMX_STRIP_REASONING": ("strip_reasoning", b),
            "CMX_AGENTIC_SEARCH": ("agentic_search", b),
            "CMX_AGENTIC_MAX_ROUNDS": ("agentic_max_rounds", int),
            "CMX_TEMPORAL_RERANK": ("temporal_rerank", b),
            "CMX_TEMPORAL_WEIGHT": ("temporal_weight", float),
            "CMX_IMPORTANCE_RERANK": ("importance_rerank", b),
            "CMX_IMPORTANCE_WEIGHT": ("importance_weight", float),
            "CMX_MULTIHOP": ("multihop", b),
            "CMX_MULTIHOP_INJECT_K": ("multihop_inject_k", int),
        }
        for key, (attr, cast) in mapping.items():
            if key in env and env[key] != "":
                try:
                    setattr(self, attr, cast(env[key]))
                except Exception:
                    pass
