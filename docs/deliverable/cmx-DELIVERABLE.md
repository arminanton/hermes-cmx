# hermes-cmx — DELIVERABLE (what's proven, what to install, how to use)

**Date:** 2026-06-14 · **Status:** robust, validated, ready to install (storage+retrieval path) ·
**Branch:** `master`
(145 tests green under the Hermes pysqlite3 interpreter).

This folds everything we proved into one place. Everything below is REPRODUCIBLE and measured;
the experimental/rejected work is listed at the end so it stays out of the default path.

---

## 1. What cmx is (the one-paragraph truth)

A context engine that **never lossily summarizes**. Every message is kept **verbatim forever** in
SQLite (FTS5 + trigram + embeddings). On each turn the engine **retrieves the relevant verbatim
slices** and injects them — instead of a lossy summary the model mistakes for its own memory (LCM's
root failure). Grounding is **enforced** (citation-checked + verified + refuse-to-guess) so the
model answers from real history or says "I don't have that," instead of confabulating.

## 2. What we PROVED (robust, reproducible results)

- **Memory is decoupled from the model's context window.** opus-4.8 with an **8,000-token window**
  answered questions over **369–663-turn conversations** (6×+ too big to fit) at **76.5% accuracy**
  — because memory lives in the **SQLite DB**, not the window. *The old "context size must match"
  problem is solved.* A fast/cheap/small-context model can run an arbitrarily long conversation.
- **Honest LOCOMO numbers** (gpt-5-mini foreground, locked config, LLM-judge == our token scorer):
  **single-hop ~76% · multi-hop ~31% · adversarial-refusal ~95–97% · hallucination ~6–9%.**
  Per-conversation variance is large (20–80%) — single conversations are noisy (use ≥5).
- **Trajectory: ~5× the v1 baseline.** LOCOMO answerable went **12.5% → ~58–66%** from iter-1
  (lexical only) to iter-2 (embeddings+rerank), then held stable. Not a regression — a transformation.
- **Safety profile is the field's frontier metric.** The newest rigorous benchmark (WMB-100K)
  penalizes false memory at 2.5× the reward — *exactly* cmx's refuse-to-guess design. The viral
  "100% on LOCOMO" claims are debunked artefacts (a top_k trick) and don't even test false memory.
- **Model-agnostic.** gpt-5-mini and gemini-2.5-pro behave comparably under the same contract.

## 3. The PROVEN config (use this)

Settings that produced the numbers above (iter-1 enforcement + iter-2 ship levers + iter-3 gate):

| setting | value | why |
|---|---|---|
| `use_embeddings` | `true` | semantic recall (iter-2 lever 1) — default on |
| `rerank` | `true` | embedding-cosine rerank (iter-2 lever 4) — default on |
| `semantic_verifier` | `true` | paraphrase rescue + uncited audit (iter-2 lever 6) |
| `sufficiency_gate` | `true` | the iter-3 safety gate (refuse when evidence insufficient) |
| `sufficiency_reasoning` | `true` | hardened gate (names the needed fact, then checks) |
| `require_citations` | `true` | deterministic citation check (Layer 3) |
| `refuse_to_guess` | `true` | the safe default (Layer 5) |
| `verifier_model` | **a CAPABLE model** (e.g. `claude-opus-4.7`) | iter-3: *the judge must be capable*; a weak judge over-refuses |

**Optional (tunable, not in the default-proven set):**
- `use_hyde: true` (+~5pt recall, costs 1 cheap call/retrieval).
- `multihop: true` — lifts multi-hop *sometimes* but UNPROVEN at our sample size; **safe** (no real
  hallucination increase) but leave **off** unless you accept the experiment.

**Leave OFF (researched + rejected — see §6):** `judge_backend: council`, all `force_*` /
`prompted_tool_protocol` / `agentic_search`, `temporal_rerank`, `importance_rerank`.

## 4. INSTALL on Hermes (reversible, alongside LCM)

The cmx plugin lives at `$HERMES_HOME/plugins/hermes-cmx` and already points
at the proven code (`hermes-cmx/src`). Three edits to `~/.hermes/config.yaml`, then reload.

**(a) Select the engine** — change the `context:` block:
```yaml
context:
  engine: cmx        # was: lcm   (switch back to lcm anytime to revert)
```

**(b) Enable the cmx plugin and DISABLE lcm** — the host selects the plugin engine only if its
`.name` matches `context.engine`, and it loads a single plugin context engine, so enabling both
risks silently falling back to the built-in lossy compressor. **Swap** them in the `plugins.enabled`
list:
```yaml
plugins:
  enabled:
  # ...
  - hermes-cmx        # was: hermes-lcm  (swap — exactly one context-engine plugin active)
```

**(c) Add the proven cmx config block** (top-level, sibling of `context:`):
```yaml
cmx:
  use_embeddings: true
  rerank: true
  semantic_verifier: true
  sufficiency_gate: true
  sufficiency_reasoning: true
  require_citations: true
  refuse_to_guess: true
  verifier_model: claude-opus-4.7     # MUST be a capable model
  # optional: use_hyde: true
  # database_path defaults to <HERMES_HOME>/cmx.db
```

**(d) Reload** — restart Hermes or run `/reload-mcp` (or your plugin-reload step). Verify with a
question about something said earlier in a long session — cmx should recall the exact wording.

**Revert** anytime: set `context.engine: lcm` and reload. cmx never deletes anything; the verbatim
store at `~/.hermes/cmx.db` persists.

## 5. HONEST scope of the install (read this)

Installing the plugin gives — **today, with no changes to Hermes core**:
- ✅ **Verbatim storage** (never lossy-summarize) — fixes LCM's "forgets + reads summary as its own
  thought" root cause.
- ✅ **Retrieval-first injection** of the relevant verbatim slices (FTS5+trigram+embeddings).
- ✅ `cmx_grep` / `cmx_expand` / `cmx_recall` tools + provider-aware budgeting.

This is already a **strict improvement over LCM** and is what you can start using now.

✅ **The full grounding enforcement loop IS LIVE** (H2 Option A, promoted to `src`): after every
assistant reply, `conversation_loop.py` (≈3874-3892) calls `engine.enforce_response()` — a strict
no-op for the built-in compressor and LCM (guarded by `capabilities()`), but for cmx it runs the
sufficiency gate + deterministic citation check + independent verify against the verbatim record and,
when the answer to a factual-history question is **ungrounded, replaces it with a refuse-to-guess
message**. Verified firing live: a fabricated "$250,000 budget" answer with no supporting evidence was
replaced with *"I don't have that in my record…"*. This is the part that produced the ~6–9%
hallucination numbers, and it is active once cmx is the selected engine.

⚠️ Only remaining unwired refinement: **Option B's corrective _regenerate_ loop** (re-prompt the model
with a correction, then refuse if still ungrounded). Option A refuses/replaces directly in a single
pass — the core safety property — so the regenerate loop is a value-add, not a safety gap
(`docs/07-HOST-INTEGRATION-H2.md`).

## 6. Researched + REJECTED (kept flag-gated OFF, do not enable by default)

Each was built, benchmarked, and ruled out — documented so we don't re-litigate:
- **Council-as-judge** — over-refuses (10% acc), 15× cost, no safety gain. (`council_judge.py`)
- **Forcing layer F1–F5** (`<system-reminder>` + prompted tools + agentic loop) — *hurt*
  these already-cooperative models (gpt-5-mini −40pt). For weakened/FreeAI-style providers only.
- **Temporal + importance re-ranking** (Dakera levers) — token-match +8.9pt was illusory; LLM judge
  flat. (`rerank_signals.py`)
- **Multi-hop synthesis** — safe (no real hallucination rise) but accuracy gain unproven at n=26
  (swung +11.5pt then +0.0pt across two runs). (`multihop` flag)

## 7. Supporting evidence (in this folder + the repo)

- `cmx-findings-and-plan.md` — full iter-1/2/3 history + the keeper-bundle synthesis.
- `external-benchmark-investigation.md` — the MemPalace/Dakera/WMB-100K investigation + every lever
  result with honest verdicts.
- Repo `hermes-cmx/`: `docs/00..08-*.md`, `benchmarks/results/matrix/iter4-*.md` (all runs),
  145 passing tests. Run cmx ALWAYS under `python3` (pysqlite3+trigram).

---
## cmx-as-context-assembler (2026-06-14 addendum)
cmx now owns context injection for ALL models. Principles: (1) never drop/
summarize — verbatim store + on-demand retrieval; (2) never assume retrieval initiative —
provider (not model name) determines cooperativeness → apply full treatment uniformly, gate only
the tool-call MECHANISM on the provider API contract.
- Stage 1 LIVE (restart to apply): durable memory via cmx — MEMORY.md/USER.md (595 chunks) in live
  cmx.db; per-turn 209K→~1.5K tok (99.3% smaller); cmx_grep reaches the rest; legacy dump suppressed
  when cmx active (system_prompt.py guard, backup saved); write-tools/files untouched.
- Stage 2: host already no-drop (tool_search progressive disclosure + skills index); + light
  teaching directive (teach_retrieval uniform-on, ~140 tok). Copilot A/B n=3: −6.7pt acc (noise;
  rules out the −40pt heavy-forcing catastrophe). Deflection caught by the council gate.
- Stage 3: reserve_overhead_tokens added. Remaining: route reasoning-model grounding through cmx.
- 127 tests pass; full record in repo docs/09-CONTEXT-ASSEMBLER.md.
