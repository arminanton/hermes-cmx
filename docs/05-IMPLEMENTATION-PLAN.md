# 05 — Implementation Plan

> Style (per the user's standing practice): **phase-gated, empirical, no time-boxing.** Each phase produces something testable; the gate to the next phase is a measured result, not a calendar date. All host (`src/`) changes happen in **isolated git worktrees** with review before promotion. cmx ships as a **plugin alongside LCM** until the eval gate passes.

## 0. Dependencies & sequencing at a glance

```
P0 skeleton ─► P1 store+retrieval ─► P2 assembly+injection ─► P3 enforcement(citations,gate)
                     │                        │                          │
                     └── needs: provider tokenizer (host)                └── P4 verifier+profiles
P2 needs: ContextEngine ABC `pre_send` hook (host)                                  │
P5 entity-graph/multi-hop (optional) ─────────────────────────────────────► P6 migrate+coexist+switch
```

Two **host-side prerequisites** gate cmx's differentiators and must land first (isolated worktrees, reviewed):
- **H1 — provider-aware tokenizer registry** (replaces single `cl100k`). Without it, budgets are wrong and the whole window-strategy is "wasted" (the user's words).
- **H2 — ContextEngine ABC extensions**: `pre_send(messages, model) -> messages` (for Layer-1 injection + Layer-2 gating), `on_tool_result(...)`, `capabilities()`, and an explicit enforcement hook for Layer 3–5. Backward-compatible no-op defaults so LCM/the default compressor keep working.

## Phase P0 — Skeleton & safe wrapper
**Build:** plugin scaffold (`plugin.yaml`, `__init__.py`, `cmx:` config loader supporting **both** `CMX_*` env and a `cmx:` YAML block), a `CmxEngine(ContextEngine)` that **delegates to current behavior** (thin wrapper, zero functional change), and a one-shot `cmx import lcm` that copies LCM's verbatim WAL into `cmx.db`.
**Gate:** loads in `hermes plugins list`; LCM's own test suite passes against the wrapper; `cmx import lcm` round-trips the store losslessly (row counts + hashes match).

## Phase P1 — Verbatim store + hybrid retrieval + tokenizer (H1)
**Build:** `messages` (append-only) + `messages_fts` + `messages_trgm` (trigram) + `chunks` + `embeddings`; the **hybrid retrieval core** (FTS5 ⊕ trigram ⊕ embeddings, RRF fusion, recency/pin priors); degrade paths (no-embeddings, no-trigram). Land **H1** (provider-aware tokenizer) in a host worktree.
**Gate:** on the identifier/code/log eval set, **Recall@5 ≥ LCM FTS + 20pts**; trigram tokenizer verified to load on the live OL8/aarch64 host (`HERMES_SQLITE_DRIVER=pysqlite3`); tokenizer counts within ±3% of provider truth on a sample per model.

## Phase P2 — Budget-aware assembly + proactive injection (needs H2.pre_send)
**Build:** the assembly pipeline (pinned sinks → recent verbatim → **retrieved evidence** → headnotes), budget-fit via H1; **Layer 1** proactive injection with labelled, **non-assistant-role**, citable evidence blocks; window strategies from `03-ADAPTIVITY.md`.
**Gate:** "evidence-present rate" (planted fact appears in the assembled prompt for its question) **≥ 95%** on the strong model and **≥ 90%** on the 32k stand-in; budget never exceeds the provider window on any test; **no assistant-role history injection** (assert in tests).

## Phase P3 — Enforcement core: forced gate + deterministic citations (Layers 2–3)
**Build:** the `cmx_*` tools (recall/expand/grep/graph_query) wired through `handle_tool_call`; **Layer 2** forced-retrieval gate (tool_choice/structured/stop-intercept per provider); **Layer 3** deterministic citation parser + SQLite span-check + bounded regenerate loop; `answer_claims` audit table.
**Gate (both strong & weak models):** **Citation validity ≥ 98%**; **Refusal-to-guess ≥ 90%** on the no-answer set; **0 phantom-citation** escapes; weak-model hallucination ≤ strong-model LCM baseline.

## Phase P4 — Independent verification + enforcement profiles (Layers 4–5)
**Build:** the **independent verifier** (cheap, separate model; deterministic path where checkable); **enforcement profiles** + the model capability registry; Layer-5 refuse-to-guess downgrade; outage fallbacks (verifier/embeddings down → deterministic-only + refuse).
**Gate:** **Hallucination rate ≤ ½ LCM** on the strong model and **≤ LCM-strong** on the weak model; **strong↔weak hallucination gap ≤ 10pts** (the agnosticism proof); failure-under-outage = **0 confabulations**.

## Phase P5 — Entity/relation graph + multi-hop (optional, additive)
**Build:** `entities`/`relations`/`mentions` extraction (schema-constrained LLM or rules) + `cmx_graph_query` (personalized-PageRank-style hop) + headnotes as navigation pointers.
**Gate:** **Multi-hop accuracy ≥ 80%**; graph extraction degrades gracefully on a cheap model (falls back to hybrid retrieval, never worse than P4).

## Phase P6 — Migration, coexistence, and the switch
**Build:** finalize `cmx import lcm`; both plugins installable; one-line switch `context.engine: cmx`; A/B per Hermes profile; docs.
**Gate (the only thing that flips the default):** full suite green vs. LCM on **both** models → user runs cmx as default in real daily workflow **≥ 3 days clean** → only then is LCM demoted (kept installed/rollback-able). If cmx loses any gate, LCM stays default and we iterate.

---

## Host (`src/agent/`) change inventory

| ID | File | Change | Risk | Why |
|---|---|---|---|---|
| H1 | `agent/tokens.py` (+ `model_metadata`) | provider-aware tokenizer registry (tiktoken/o200k for OpenAI, anthropic tokenizer, gemini, heuristic fallback) | low–med | correct budgets; without it the window strategy is wasted |
| H2a | `agent/context_engine.py` | add optional `pre_send(messages, model)`, `on_tool_result(...)`, `capabilities()`, enforcement hook; **no-op defaults** | low | enables Layers 1–5 without breaking LCM/default |
| H2b | `agent/conversation_compression.py` | replace `try/except TypeError` kwargs probing with `engine.capabilities()` | low | clean, explicit engine contract |
| H2c | `agent/conversation_loop.py` | call `pre_send`/`on_tool_result` at the mapped sites; leave existing `_compress_context` sites intact | med | wire the hooks; isolated + reviewed |
| — | `agent/context_compressor.py` | **no change** | — | default engine untouched; cmx is opt-in via config |

All host changes: isolated worktree → tests green → smoke on each provider → explicit user ack before promotion (the workflow we used for the LCM patches).

## Configuration surface (initial)

```yaml
context:
  engine: cmx                 # the one-line switch (default stays 'lcm' until P6 gate)
cmx:
  database_path: ""           # default <HERMES_HOME>/cmx.db
  retrieval:
    use_trigram: true
    use_embeddings: true      # degrade to false if index unavailable
    embedding_model: ""       # provider/model; empty = disabled
    fusion: rrf
    k_default: 8
  enforcement:
    require_citations: true
    forced_gate: auto         # auto = factual-history turns only
    verifier_model: "gpt-5-mini"
    verifier_mode: profile    # profile = strict→mandatory, light→sampled
    refuse_to_guess: true
  models: { ... }             # capability registry (see 03-ADAPTIVITY.md)
```
Also honored as `CMX_*` env vars (e.g. `CMX_VERIFIER_MODEL`), mirroring LCM's discipline but **fixing** the missing-YAML-block gap.

## Test strategy

- **Unit:** retrieval fusion, tokenizer per provider, citation parser/span-check, profile selection, budget allocator.
- **Property/deterministic:** "no assistant-role history injection," "never exceed window," "no shipped uncited factual claim survives," "outage ⇒ refuse not confabulate."
- **Integration:** full turn lifecycle on a live `hermes -z` session per provider (strong + weak), mirroring the smoke test we used for the LCM patches.
- **Eval:** the `04-EVALUATION.md` suite as the phase gates.

## Definition of done (project)
cmx passes every `04` gate on a strong **and** a weak model, runs ≥ 3 days as the user's daily default with zero confabulation incidents, and `cmx import lcm` makes the switch reversible — at which point cmx becomes the default context engine and LCM is retained only for rollback.
