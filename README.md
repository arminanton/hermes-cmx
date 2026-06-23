# hermes-cmx — Context Memory eXchange

> A retrieval-first context engine for [Hermes Agent](https://github.com/NousResearch/hermes-agent),
> and a drop-in successor to **hermes-lcm**. It **never lossily summarizes**: it keeps
> **100% verbatim history** (SQLite FTS5 + trigram + embeddings by default; Postgres optional),
> **retrieves** the exact relevant slices on every turn, and **enforces grounding** so the model
> answers from real history or honestly refuses — regardless of which model it is or how small its
> context window.

cmx installs as a Hermes context-engine plugin (`context.engine: cmx`). The grounding enforcement
(H2) runs in-host after every reply: it checks the answer against the verbatim store and replaces
an ungrounded answer with a refusal. 145 tests pass under the Hermes `pysqlite3` interpreter. See
[`docs/deliverable/cmx-DELIVERABLE.md`](docs/deliverable/cmx-DELIVERABLE.md) for the proven config
and install steps.

**Supersedes:** `hermes-lcm` (kept as a one-config-line revert) · **License:** MIT

---

## Why cmx is better than LCM (measured, not asserted)

LCM is **compress-and-hope**: at a threshold it replaces older turns with **lossy summaries**,
injects them as if the assistant wrote them, and *hopes* the model calls `lcm_expand` to recover
detail. Our June-2026 re-assessment traced the user-reported "forgets + hallucinates" to exactly
this — the model reads its own lossy paraphrase, treats it as truth, and confabulates. cmx
removes that dependency: there are no summaries to misread, and retrieval is the engine's job,
not the model's.

**Head-to-head on identical LOCOMO questions** (gpt-5-mini, constrained window —
`benchmarks/results/vs-lcm-gpt-5-mini-low.md`):

| metric | hermes-lcm | **hermes-cmx** |
|---|---|---|
| answerable accuracy | 58.3% | **66.7%** |
| adversarial refusal | 83.3% | **100%** |
| **hallucination (shipped)** | 10.0% | **0.0%** |

| dimension | hermes-lcm | hermes-cmx |
|---|---|---|
| Working-context strategy | lossy summary DAG **replaces** verbatim | **no lossy summary** — verbatim slices retrieved on demand |
| Who decides to retrieve | the **model** (often won't) | the **engine** (always; model can't opt out) |
| Grounding | none — model may guess | **enforced**: proactive injection → forced pre-answer gate → deterministic citation check → independent verify → refuse-to-guess |
| Retrieval | FTS5 (word tokenizer) + LIKE | **hybrid FTS5 + trigram + embeddings**, fused + reranked |
| Tokenizer | single `cl100k` for all models | **provider-aware** per-model budgets |
| Survives session-id rotation | ❌ loses history when Hermes rotates the id on compaction/checkpoint | ✅ **lineage normalization** — one logical conversation keeps one effective id |
| Model / reasoning switch mid-chat | n/a | ✅ safe — store is independent of the model window |
| Weak models (gpt-5-mini) | degrade badly | **same grounding contract**, stricter enforcement profile |

---

## The one-paragraph pitch

A frontier LLM over a closed API has a **finite** context window, so "infinite context where the
model literally sees everything" is physically impossible. What cmx delivers instead is
**lossless recoverable infinity**: every message is stored verbatim forever in the store, nothing is
paraphrased away, and on every turn the engine retrieves the exact relevant slices and
forces + verifies that the model answers from them. Memory lives in the **database, not the
model's window** — proven: opus-4.8 with an **8k-token window** answered questions over
**600-turn** conversations. So a fast, cheap, small-context model can run an arbitrarily long
conversation; only the grounding *judge* needs to be capable.

---

## What changed since the first cut (the improvements added)

- **H2 grounding enforcement is live in-host.** After every reply, Hermes calls the engine's
  `enforce_response`; for cmx it runs the sufficiency gate + citation check + verify and
  **replaces an ungrounded answer with a refusal** (no-op for LCM/compressor). Verified live.
- **Robust across Hermes session-id rotation** (the big reliability fix). Compaction and
  checkpoints rotate the `session_id`; cmx maps every rotated id to a lineage **root**, so
  retrieval never loses the prior conversation. (`store.py` `session_lineage`,
  `examples/02_survives_session_rotation.py`.)
- **Thread-safe store.** The SQLite connection is opened `check_same_thread=False` and serialized
  with a lock, so the engine is safe to use from the host's per-turn worker threads
  (`asyncio.to_thread`) and during compaction — a default connection would crash with a
  thread-affinity `ProgrammingError`. (`db.py` `_LockedConnection`, `tests/test_thread_safety.py`.)
- **Content-hash dedup ingest** — survives the host swapping the message list across compaction
  (no missed/duplicated turns).
- **Model / reasoning switches are safe** — `update_model` refreshes the window/tokenizer; the
  verbatim store is untouched.
- **Honest measurement.** We built an LLM-judge scorer to check our token-match scorer (they
  agree — no lenient-judge inflation) and re-based every claim on pooled LOCOMO runs.
- **Every accuracy lever was tried and judged** — Council-as-judge, a CC/MaxAI forcing layer,
  temporal/importance re-ranking, multi-hop synthesis. Most were **rejected on honest evidence**
  and kept flag-gated **off**. Full ledger: [`benchmarks/README.md`](benchmarks/README.md).

---

## Honest scope (read this before trusting it blindly)

- **Normal answerable recall is strong** — evidence-recall@8 **55% → 82%** from the v1 lexical
  baseline to the iter-2 ship stack.
- **Hallucination is mitigated, not "solved".** On *deliberately adversarial,
  on-topic-but-unanswerable* questions (the hardest, rarest type), larger samples show **5–31%**
  residual depending on model — the early "0%" was small-sample noise, and we report it openly
  (`benchmarks/results/iter3-HARDENING-conclusion.md`). refuse-to-guess minimizes it; it is not
  a zero guarantee on adversarial. For critical work, keep reasoning on and prefer
  answerable-only flows.
- **Pool ≥5 conversations.** Per-conversation variance is 20–80%; single runs are noise.

---

## Try it (no model needed for the first two)

```bash
git clone https://github.com/arminanton/hermes-cmx
cd hermes-cmx
PYTHONPATH=src python3 examples/01_store_and_retrieve.py
PYTHONPATH=src python3 examples/02_survives_session_rotation.py
```
See [`examples/`](examples/) for grounded-answer and refuse-to-guess demos too.

> **SQLite needs the trigram tokenizer.** cmx uses SQLite's FTS5 `trigram` tokenizer (SQLite
> ≥ 3.34). Hermes ships a `pysqlite3` build (SQLite 3.53) that has it, so run cmx under the Hermes
> venv python in production. For the standalone examples, any Python whose `sqlite3` has FTS5 +
> trigram works; cmx degrades to FTS5-only word matching if trigram is unavailable.

## Install as a Hermes plugin

Drop the repo into your Hermes plugins directory and select it as the context engine:

```bash
git clone https://github.com/arminanton/hermes-cmx "$HERMES_HOME/plugins/hermes-cmx"
```
```yaml
# in $HERMES_HOME/config.yaml
context:
  engine: cmx
```

The default backend is **SQLite** at `$HERMES_HOME/cmx.db` (zero-config). To run on **Postgres**
(pgvector + pg_trgm) instead, set `cmx.backend: postgres` and a DSN via `cmx.pg_dsn` (or the
`CMX_PG_DSN` env); see [`deploy/postgres/`](deploy/postgres/) for a ready-to-run container.

---

## Repo map

```
hermes-cmx/
├── README.md                            ← you are here
├── LICENSE                              MIT
├── plugin.yaml __init__.py              Hermes plugin manifest + entrypoint
├── examples/                            4 runnable demos (+ README)
├── src/cmx/
│   ├── db.py store.py retrieval.py tokenizer.py   storage + hybrid FTS5/trigram/embeddings + provider tokenizer
│   │                                              (store.py adds session_lineage: rotation-proof retrieval)
│   ├── pg_store.py store_factory.py substrate.py  optional Postgres backend (pgvector + pg_trgm) + selector
│   ├── embeddings.py rerank.py rerank_signals.py  semantic recall + cosine rerank (+ rejected temporal/importance)
│   ├── profiles.py assembly.py                    per-model capability profiles + budget-aware injection
│   ├── enforcement.py llm.py engine.py            5-layer grounding + sufficiency gate + model client + turn loop
│   ├── council_judge.py forcing.py tool_dialects.py   researched + REJECTED levers (flag-gated off)
│   ├── hermes_engine.py                           Hermes ContextEngine shim (lineage, dedup, enforce_response)
│   └── migrate.py                                 lossless `import lcm`
├── tests/                               145 tests (incl. e2e grounding + session-lineage)
├── benchmarks/                          runners + README (lever ledger) + results/ (every measured run)
├── deploy/postgres/                     optional Postgres backend (compose + verify script)
└── docs/
    ├── 00..06-*.md                      overview, architecture, enforcement, adaptivity, eval, plan, risks
    ├── 07-HOST-INTEGRATION-H2.md        the in-host enforcement hook (Option A live; Option B = optional)
    └── deliverable/                     cmx-DELIVERABLE.md (proven config + install), INTEGRATION-AUDIT.md,
                                         external-benchmark-investigation.md
```

## Read order

1. `docs/00-OVERVIEW.md` — *why*, and the design principles.
2. `docs/01-ARCHITECTURE.md` — storage, retrieval, turn lifecycle.
3. `docs/02-GROUNDING-ENFORCEMENT.md` — the heart: how the engine forces grounding.
4. `benchmarks/README.md` — every lever we tried and the honest numbers.
5. `docs/deliverable/cmx-DELIVERABLE.md` — the proven config + how it's installed.
6. `docs/deliverable/external-benchmark-investigation.md` — the MemPalace/Dakera/WMB-100K reality check.

## Non-negotiable principles (the contract)

1. **Verbatim is the only truth.** The model reasons from retrieved original text, never a paraphrase.
2. **The engine owns grounding, not the model.** No part of the design depends on the model *choosing* to behave.
3. **Every factual claim is checkable.** Cited → verified against the verbatim store deterministically. Uncited → caught by an independent verifier or refused.
4. **Model-agnostic.** The same contract holds for gpt-5.5 and gpt-5-mini; only the *enforcement strictness* changes.
5. **Window-aware.** The engine sizes everything to the model's real token budget; a small window tightens retrieval and leans harder on verification — it never silently overflows.
6. **Lossless and reversible.** Nothing is ever deleted; cmx can import LCM's store, and reverting is one config line.
7. **Measured, not asserted.** Claims trace to a result file; rejected levers stay rejected and documented.
