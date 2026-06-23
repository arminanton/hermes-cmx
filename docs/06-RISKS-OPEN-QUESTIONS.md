# 06 — Risks & Open Questions

## 1. Risks (with mitigations)

### R1 — Retrieval precision is the floor of everything
If the right verbatim slice isn't surfaced, Layer 1 injects the wrong thing and Layers 3–4 can only downgrade to "I don't know." **This is the dominant risk** and where most engineering effort goes.
- *Mitigations:* hybrid fusion (FTS5 ⊕ trigram ⊕ embeddings) so three failure modes must coincide to miss; query expansion from extracted entities/identifiers; chunk-level granularity; the ablation evals quantify the floor so we know exactly how much precision each accelerator buys. Worst case degrades to **refuse**, never to confident-wrong (Scenario D).

### R2 — Trigram FTS5 on OL8 / aarch64
The `trigram` tokenizer needs a recent SQLite+FTS5. The host already uses `HERMES_SQLITE_DRIVER=pysqlite3`; there's a known `sqlite-trigram-ol8-aarch64` runbook.
- *Mitigations:* P1 gate explicitly verifies the trigram tokenizer loads on the live host before anything depends on it; documented fallback chain is **trigram → FTS5-word + LIKE** (LCM's floor). cmx never *requires* trigram, it *prefers* it.

### R3 — Verification cost & latency (Layer 4)
An extra model call per factual-history turn adds cost/latency; the user's "wasting the implementation" concern includes overhead.
- *Mitigations:* verifier is a **cheap, separate** model (gpt-5-mini); runs **only on factual-history turns**; **sampled/async** on Light profiles; deterministic checks preferred whenever a claim references a store id/identifier (no LLM call at all); per-turn budget ceiling with graceful fallback to deterministic-only.

### R4 — Embeddings dependency & index size
`sqlite-vec`/`sqlite-vss`/faiss adds a runtime dep and disk; aarch64 builds can be painful.
- *Mitigations:* embeddings are **optional and degradable** (FTS5+trigram still strong for the identifier-heavy workload); ship as opt-in; lazy/async embedding writes so ingest never blocks; cap/evict cold vectors (demote, never delete the verbatim row).

### R5 — Provider variance in forced-tool support (Layer 2)
`tool_choice=required` / structured-output semantics differ across Claude/OpenAI/Gemini and local models.
- *Mitigations:* the gate has three implementations (forced-tool → structured two-step → stop-sequence interception); `capabilities()` + the model registry pick the strongest available per provider; `_default` assumes the weakest and uses interception.

### R6 — "Factual-history turn" classifier errors
Mis-classifying a turn as non-factual skips enforcement; over-classifying adds overhead.
- *Mitigations:* bias the classifier toward **forcing when unsure** (false positives cost latency, false negatives cost grounding); cheap/deterministic signals first (question words, references to "we/earlier/you said/the X we…", presence of identifiers); measured by the refusal/hallucination evals.

### R7 — Latency from synchronous indexing
Trigram/embedding/entity extraction at ingest could slow turns.
- *Mitigations:* index **asynchronously** below a freshness threshold (LCM's "zero-cost continuity" idea); the recent verbatim window covers the just-ingested turns until the index catches up; never block a turn on embedding writes.

### R8 — Reversed / superseded facts
A decision later overturned must not be retrieved as current truth.
- *Mitigations:* recency prior in fusion; entity-graph can mark superseded relations; eval set includes **reversal traps**; when both old and new appear, inject **both** with timestamps and let the (now-grounded) model reconcile, citing the latest.

### R9 — Scope creep vs. LCM's maturity
LCM has years of hardening (lifecycle, circuit breakers, doctor, externalization). A rewrite risks regressing on robustness.
- *Mitigations:* **reuse** LCM's proven plumbing (P0 wrapper, externalizer, lifecycle); cmx changes the *memory contract*, not the *operational* hardening; coexistence + eval-gate means we never ship a regression as default.

### R10 — Verifier/foreground correlated errors
If the verifier shares the foreground model's blind spots, it may rubber-stamp a hallucination.
- *Mitigations:* prefer a **different model family** as verifier; give it only `(claim, evidence)` in isolation (no leading context); prefer deterministic checks; measure verifier precision/recall directly in evals.

## 2. Open questions (decisions to make, with a recommended default)

| # | Question | Options | Recommended default |
|---|---|---|---|
| Q1 | Embeddings in v1, or FTS5+trigram first? | (a) include now (b) defer to P5 | **(b) defer** — prove FTS5+trigram floor first; add embeddings if recall@k gate not met |
| Q2 | Verifier model | gpt-5-mini / gemini-flash / deterministic-only | **gpt-5-mini default**, deterministic where checkable, configurable |
| Q3 | Forced gate default scope | all turns / factual-history only | **factual-history only** (auto), overridable to all |
| Q4 | Keep LCM's summary DAG at all (as optional headnotes)? | drop entirely / keep as *non-prompt* navigation only | **non-prompt headnotes only** — never reasoned-from |
| Q5 | New repo location / name | `hermes-cmx` here / GHE fork / rename | **`hermes-cmx` here**; push to user's GHE when P0 lands |
| Q6 | Citation format | inline `[id=N]` / structured JSON tail / both | **inline `[id=N]`** for fluency + a structured tail on Strict profile |
| Q7 | Entity graph (P5) — build or skip? | build / skip unless multi-hop gate fails | **gate-driven** — only if P4 multi-hop < 80% |
| Q8 | Default-switch authority | auto on green / user-confirmed | **user-confirmed** after ≥3-day dogfood (per standing rule) |

## 3. Explicitly out of scope (restating non-goals)
- No model training/fine-tuning (closed APIs).
- No attempt to make the model literally see all tokens (physically impossible).
- Not a general RAG library — scoped to Hermes conversational/agentic memory.
- No removal of LCM until the eval gate + dogfood pass.

## 4. First action when we start building
Land **P0** (skeleton + safe wrapper + `cmx import lcm`) and the **H1 provider-aware tokenizer** in an isolated host worktree — these are independent, low-risk, and unblock everything. Everything else is gated on measured retrieval precision (P1) and the host `pre_send` hook (H2/P2).
