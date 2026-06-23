# cmx benchmarks, levers & results

Everything we measured while building cmx — every accuracy/safety "lever" we tried, why we
kept or rejected it, the honest numbers, and how to reproduce them. Nothing here is
aspirational; every verdict below is backed by a result file in [`results/`](results/).

> **Run everything under the Hermes interpreter** (it ships `pysqlite3` = SQLite 3.53 with the
> trigram tokenizer; system Python lacks it and 2 store tests will fail):
> ```bash
> PYTHONPATH=src python3 benchmarks/<runner>.py
> ```

---

## TL;DR — the honest headline

- **cmx beats LCM head-to-head** on the same LOCOMO questions (gpt-5-mini, constrained window):

  | metric | hermes-lcm | **hermes-cmx** |
  |---|---|---|
  | answerable accuracy | 58.3% | **66.7%** |
  | adversarial refusal | 83.3% | **100%** |
  | **hallucination (shipped)** | 10.0% | **0.0%** |

  *(`results/vs-lcm-gpt-5-mini-low.md`)*

- **Normal answerable recall is strong.** Evidence-recall@8 went **55% → 82%** from the
  lexical v1 baseline to the iter-2 ship stack (`results/iter2-FINAL-assessment.md`).
- **Hallucination is mitigated, not "solved".** On *deliberately adversarial,
  on-topic-but-unanswerable* questions (LOCOMO category 5 — the hardest, rarest type), larger
  samples show **5–31%** residual depending on model. The early "0%" was small-sample noise;
  we report this openly (`results/iter3-HARDENING-conclusion.md`). The refuse-to-guess design
  minimizes it, but does not guarantee zero. **Do not claim 0% on adversarial.**
- **Per-conversation variance is huge (20–80%).** Always pool **≥5 LOCOMO conversations**;
  single-conversation numbers are noise (this is also Dakera's published lesson, and we
  reproduced it on our own runs).

---

## The lever ledger (everything we tried)

Each lever was implemented one-at-a-time in an isolated branch and measured. ✅ = in the
shipped default, ☑️ = implemented & opt-in, ❌ = rejected (kept flag-gated **off** so we
don't re-litigate), ⚠️ = unproven.

### Iteration 1 — enforcement core (the foundation, all kept)
Verbatim SQLite store · FTS5 + trigram + embeddings hybrid retrieval · 5-layer grounding
(proactive injection → forced pre-answer gate → deterministic citation check → independent
verify → refuse-to-guess) · inject cap=4 · provider-aware tokenizer · per-model profiles.

### Iteration 2 — retrieval & safety levers (`results/iter2-*`)
| # | Lever | Verdict | Evidence |
|---|---|---|---|
| 1 | Embeddings (semantic kNN) | ✅ **ship, default-on** | recall@8 55→65%; degrades to lexical if no embedder |
| 2 | Tool loop (model-driven retrieval) | ❌ **reject (hard)** | adversarial-refusal→0%, halluc→27–39% even with the verifier |
| 3 | HyDE query expansion | ☑️ **ship, opt-in** | recall@8 +10–15pt; 1 cheap call/retrieval |
| 4 | Embedding-cosine rerank | ✅ **ship, default-on** | recall@8 +12pt (82% full stack) |
| 5 | Higher inject_k for weak models | ❌ **reject** | no accuracy gain + breaks the guardrail; cap=4 is a feature |
| 6 | Semantic verifier (NLI judge) | ☑️ **ship, opt-in** | paraphrase rescue + uncited-claim audit; **needs a capable judge** |

**Iter-2 lesson:** let the *engine* retrieve, never the model. Lever 2 (handing retrieval to
the model) was the single biggest safety regression.

### Iteration 3 — driving the residual hallucination down (`results/iter3-*`)
| Lever | Verdict | Evidence |
|---|---|---|
| L1 Abstain on low retrieval-confidence | ❌ reject | answerable vs adversarial max-cosine *overlap* (0.59 vs 0.57) — a similarity threshold can't catch on-topic-but-absent |
| **L2 Evidence-sufficiency pre-gate (binary)** | ✅ **KEEP ★** | with a capable judge: 66.7%/100%/0% vs baseline 66.7%/83.3%/11.8% — the semantic "is this actually answerable?" check |
| L3 Chunk-level embeddings | ❌ reject | whole-turn embeddings already ~100% recall on facts buried in ~8k-char turns |
| L4 LLM-relevance reranker | ☑️ opt-in | recall@8 80→90; +1 model call/retrieval |
| L5 Self-consistency refusal | ☑️ opt-in | adversarial 83→92%, halluc 11.8→6.2%; N× generation cost |
| L6 Graded sufficiency confidence | ❌ reject | binary YES/NO beat a 0–100 score (graded adds noise) |

**Iter-3 lessons:** (1) the residual is *on-topic-but-unanswerable* — only a **semantic
sufficiency check** (L2) gates it, not similarity (L1). (2) **Judge strength matters for the
gate** (a weak judge over-refuses to ~56% accuracy; a capable judge holds 100% refusal at full
accuracy) — but is irrelevant for citation *rescue*. (3) **Less is more** — stacking more
levers (L4 aggressive feeding, hard citation contracts) made hallucination *worse*.

### Iteration 4 — multi-judge & forcing exploration (`results/matrix/iter4-*`)
| Lever | Verdict | Evidence |
|---|---|---|
| Forced pre-answer gate (ablation) | ✅ **mandatory** | removing it (verify-only) **doubled** hallucination to ~44% — a post-hoc check can't catch a confident wrong answer |
| Multi-persona **Council** as the judge | ❌ **reject** | over-refused to **10%** answerable accuracy (vs 70% single-opus gate), **15× cost** (4127s vs 271s), no safety gain |
| Single capable model as the gate | ✅ **ship** | this is the locked judge config (`verifier_model: claude-opus-4.7`) |
| Forcing layer F1–F5 (CC/MaxAI `<system-reminder>` + prompted tools + agentic loop) | ❌ **reject** | **−40pt** on these already-cooperative copilot models; only helps *weakened*/FreeAI-style providers |

### Post-iteration — Dakera-inspired & multi-hop (`results/` + `docs/deliverable/external-benchmark-investigation.md`)
| Lever | Verdict | Evidence |
|---|---|---|
| Temporal re-ranking (recency) | ❌ reject | token-match +8.9pt was **illusory**; LLM-judge flat |
| Importance re-ranking | ❌ reject | flat under LLM-judge |
| Multi-hop synthesis (lever 2) | ⚠️ **unproven** | safe (no real hallucination rise) but accuracy swung **+11.5pt then +0.0pt** across two runs at n≈26 — noise-dominated; leave **off** unless you accept the experiment |

---

## External-benchmark reality check

We investigated the headline claims circulating in mid-2026 (full write-up:
`docs/deliverable/external-benchmark-investigation.md`):

- **"100% on LOCOMO" (MemPalace) — debunked.** The score is a `top_k > #sessions` retrieval
  artefact; it doesn't even test false memory. Official LOCOMO F1 gives *GPT-4 with full
  context* only ~32%.
- **Dakera (88%)** uses an LLM-judge over the full 1540-question set. Its real lesson —
  **pool many conversations, single ones are noise** — we adopted and reproduced.
- **WMB-100K validates cmx's core bet.** Its scoring penalizes false memory at **2.5× the
  reward for a correct one** — i.e. a confident wrong answer is far worse than a refusal.
  That is exactly cmx's refuse-to-guess design.
- **Our scorer is honest.** We built an LLM-judge scorer and re-scored: it agrees with our
  token-match scorer (no lenient-judge inflation).

---

## Benchmark runners (what each script does)

| Runner | What it measures |
|---|---|
| `synthetic.py`, `run_eval.py` | synthetic planted-fact suite (deterministic recall + grounding) |
| `run_real_eval.py` | grounding contract on 4 real models → `results/real-models.md` |
| `run_vs_lcm.py` | **head-to-head cmx vs LCM** on LOCOMO → `results/vs-lcm-*.md` |
| `run_locomo_eval.py` | the main LOCOMO harness (answerable + adversarial + hallucination), all lever flags + LLM-judge scorer |
| `run_lever_bench.py` | temporal/importance re-rank levers |
| `run_multihop_bench.py` | multi-hop synthesis lever (+ fair judge-halluc metric) |
| `run_diagnostic.py` | window-decoupling & single- vs multi-hop diagnostic |
| `run_council_matrix.py`, `run_finalists.py`, `run_gate_sweep.py`, `run_lock_confirm.py` | iter-4 Council/judge matrix → `results/matrix/iter4-*` |
| `run_forcing_bench.py`, `run_forcing_ablation.py` | forcing-layer F1–F5 ablation |
| `_judge_ablation.py` | judge-strength ablation → `results/judge-ablation-*` |
| `recall_diag.py`, `recall_chunks.py`, `calib_abstain.py` | retrieval-recall & abstain-calibration diagnostics |
| `live_inhost_test.py` | **live in-Hermes-process** test (plugin → compaction injection → real model cites) |
| `analyze_matrix.py` | aggregates matrix JSONL into the summary tables |

See [`results/README.md`](results/README.md) for a map of every result file.
