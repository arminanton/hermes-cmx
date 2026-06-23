# 00 — Overview: Problem, Diagnosis, Philosophy

## 1. The problem we are solving

The user runs Hermes with a context handler (hermes-lcm) and reports two symptoms on long sessions:

- **Forgetting** — facts stated earlier become unavailable later.
- **Hallucination** — the model answers questions about its own history with confident, wrong detail.

The naive reading is "compression is too aggressive." The real cause is deeper and is what cmx is designed around.

## 2. Diagnosis (from the June 2026 re-assessment of hermes-lcm v0.14.0)

A full re-assessment verified, against live code, *why* LCM produces those symptoms. The findings that shape cmx:

1. **Lossy summaries replace verbatim in the prompt.** Past `_effective_assembly_token_cap` LCM swaps older turns for summary-DAG nodes. Multi-level condensation compounds loss (~0.2 → 0.4 → 0.4 budget ratios). The model can no longer see what was actually said.
2. **Summaries are injected as `assistant` role** (`engine.py:3957`). The model reads a lossy paraphrase of a *user* turn — or of a *rejected* decision — *as if it were its own prior thought*, then elaborates from it. This is the strongest mechanistic driver of "hallucinating prior decisions."
3. **The model is trusted to drill back and won't.** LCM provides `lcm_expand`/`lcm_grep`, but using them is the model's choice. With a confident summary in front of it, the model guesses instead of retrieving. The drill-down instruction was even dropped after the first compaction (`compression_count == 0` gate) — fixed in our interim patch, but the *dependence on model goodwill* remains architectural.
4. **On summarizer outage, L3 stored raw head/tail fragments as a "summary"** with an empty expand hint — fragments read as a coherent narrative. (Interim-patched in LCM.)
5. **One `cl100k` tokenizer for all providers** → thresholds fire wrong on Claude/Gemini → off-budget summaries → more fallback loss.
6. **Retrieval is FTS5 word-tokenizer + LIKE fallback only** — weak on identifiers, substrings, typos; no trigram, no embeddings. The substrate for "retrieve the exact slice" is under-powered.

**The throughline:** LCM is a *compress-and-hope* design. It compresses the working set (losing fidelity) and hopes the model recovers it (it doesn't). cmx removes both halves of that bet.

### What we already shipped to LCM (interim, not the rewrite)

To stop the bleeding now, we applied to the live LCM plugin: a strong summarizer model, retain-all across `/new`, deeper condensation, a durable decision log, less-frequent compaction, plus two code patches (persist the drill-down note; replace the L3 fragment with a labelled placeholder + recovery pointer). These make the *hybrid* more reliable. cmx is the *architecture* change.

## 3. Why physics forces a specific design

A closed-API model (Claude / GPT / Gemini) has a **finite** context window. Therefore:

- "Infinite context where the model always sees everything" is **impossible**. Something must always choose the window contents.
- The achievable maximum is **lossless recoverable infinity**: keep everything verbatim forever; on each turn, put the *relevant* verbatim in the window and *guarantee* the model uses it.
- Even huge windows (1M) suffer **"lost in the middle"** (Liu et al. 2023) and *context rot* (cited by the LCM paper itself) — so "just use a big window" is not grounding either. Retrieval + verification still matter at 1M.

This is why cmx is **retrieval-first** (no lossy summary) and **enforcement-centric** (the engine guarantees grounding).

## 4. Design philosophy

### 4.1 Engine-managed determinism, taken all the way
The LCM paper's own thesis is that an *engine* with deterministic primitives beats a *model* improvising its own memory (its "structured programming vs GOTO" argument against RLM's full model autonomy). cmx agrees — and applies the same principle to **grounding**: the engine, not the model, decides what evidence is present and whether an answer is supported. We do **not** adopt the "give the model agency / let it self-critique" pattern (Self-RAG, MemGPT paging as the primary mechanism), because it re-introduces dependence on the model's cooperation — the exact failure we are eliminating.

### 4.2 Retrieval-first, summary-never
The model reasons only from **retrieved verbatim**. cmx may maintain *structural* indexes (entities/relations, embeddings, short navigational "headnotes") — but these are **accelerators that point at verbatim**, never paraphrases the model treats as truth. The line: *an index helps you find the original; it never substitutes for it in the model's reasoning.*

### 4.3 Grounding is enforced, not requested
Five engine-side layers (see `02-GROUNDING-ENFORCEMENT.md`) make grounding independent of the model's choices: proactive injection, forced retrieval gate, deterministic citation check, independent verification, refuse-to-guess default.

### 4.4 Model-agnostic, window-aware
The same contract holds for any model; a **per-model enforcement profile** tunes strictness, and a **provider-aware tokenizer** sizes every allocation to the model's real window (see `03-ADAPTIVITY.md`).

## 5. Goals / Non-goals

### Goals
- G1. **No lossy summarization** of working context. Ever.
- G2. **Measurably lower hallucination** than LCM on both a strong and a weak model.
- G3. **Refusal-to-guess** when evidence is absent (no confident confabulation).
- G4. **High-precision retrieval** over identifiers/code/logs (trigram + FTS5 + embeddings).
- G5. **Correct behavior on small-context models**, not just 1M-token models.
- G6. **Lossless + reversible**: import LCM's store; coexist; switch default only after the eval gate.
- G7. **Deterministic, auditable grounding**: every shipped factual claim is traceable to a verbatim row.

### Non-goals
- N1. Making the model "see everything at once" (impossible; not attempted).
- N2. Beating LCM on raw token-efficiency of the prompt (we pay for indexes + verification; we trade tokens for correctness, within a bound).
- N3. Training or fine-tuning models (closed APIs; prompt + engine only).
- N4. A general-purpose RAG library (cmx is a Hermes context engine, scoped to conversational/agentic memory).
- N5. Removing LCM on day one (parallel until evals pass).

## 6. What we keep vs. drop from LCM

**Keep (proven, reuse or port):**
- Immutable, append-only **SQLite verbatim store** — the foundation LCM got right.
- **FTS5** full-text substrate (extend with trigram + embeddings).
- **Externalization** of oversized tool outputs to disk with compact refs.
- **Lifecycle state machine** (session start/end/reset/rollover) and **circuit breaker** patterns.
- **Doctor / schema-repair** tooling philosophy.
- Env-var configuration discipline (cmx will *also* support a real config block — see plan).

**Drop / replace (the cause of the symptoms):**
- The **lossy summary DAG as prompt content** → replaced by retrieval of verbatim slices.
- **Assistant-role injection** of historical context → replaced by clearly-labelled engine-authored evidence blocks (system/tool role, never "your words").
- The **single `cl100k` tokenizer** → provider-aware registry.
- **Model-discretion retrieval** → engine-enforced grounding.

## 7. Relationship to prior research

- **LCM (Ehrlich & Blackman, arXiv 2605.04050)** — adopt its engine-managed determinism and dual-state storage; reject its lossy-summary-as-prompt mechanism.
- **RLM (Recursive Language Models, Zhang et al.)** — adopt its insight that context management is an *active* process over an external store; reject full model autonomy (production stochasticity). cmx sits between LCM and RLM: engine-driven retrieval over verbatim, with the model as a constrained consumer.
- **HippoRAG / RAPTOR / Mem0 / Self-RAG / StreamingLLM** — mined for the entity-graph (multi-hop), pinned-sinks, and critique *ideas*, but every "model decides" pattern is converted to "engine enforces."
- **OOLONG / LOCOMO / "Lost in the Middle" / RULER** — evaluation references (see `04-EVALUATION.md`).
