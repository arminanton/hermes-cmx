# Iteration 2 — FINAL Re-Assessment

Six ranked retrieval/safety "levers" implemented one-at-a-time in isolated worktree
`iter2`, each empirically tested (deterministic recall@k for retrieval levers; real-model
LOCOMO gate at milestones). Hard guardrail throughout: hallucination ≈0%.

## Verdicts
| Lever | Verdict | Reason |
|---|---|---|
| 1 Embeddings (semantic kNN) | **SHIP (default-on)** | recall@8 55→65%; degradable to lexical |
| 2 Tool loop (model-driven retrieval) | **REJECT (hard)** | adversarial-refusal→0%, halluc→27-39%, even WITH lever 6 |
| 3 HyDE query expansion | **SHIP (opt-in)** | recall@8 →75% (+10-15pt); 1 cheap call/retrieval |
| 4 Embedding-cosine rerank | **SHIP (default-on)** | recall@8 →77% (+12pt), 82% full stack |
| 5 Higher inject_k for weak models | **REJECT** | no acc gain + breaks guardrail; cap=4 is a feature |
| 6 Semantic verifier (NLI judge) | **SHIP (opt-in)** | paraphrase rescue + uncited audit; needs a CAPABLE judge |

## Ship config = embeddings + HyDE + rerank + cap-4 + semantic-verifier, NO tools
- **Cumulative evidence-recall@8: 55% (v1 lexical) → 82%** (recall@15 85%).
- **opus-4.7:** 66.7% acc, **100%** adversarial refusal, **0%** hallucination.
- **gpt-5-mini + capable judge (n=30):** 61.1% acc, 91.7% adversarial refusal, **6.2%**
  hallucination — **2.5–5× the v1 accuracy (12.5-25%)** at low residual confabulation.

## Load-bearing findings
1. **Precision > recall for weak models.** v1 over-refusal was a retrieval-precision
   problem (levers 1/3/4), not an inject-count problem. More evidence (lever 5) →
   more confabulation, not more answers.
2. **Engine-driven injection + refuse-to-guess is safe; model-driven retrieval is not.**
   No verifier can tell a well-cited *wrong* answer from a correct one. Lever 2 rejected.
3. **The judge must be capable.** Weak-judges-weak is the residual-hallucination weak
   link. Set `verifier_model` ≥ the foreground tier.

## Production wiring (worktree iter2, NOT promoted)
`CmxContextEngine` now builds the embedder (lever 1+4), optional HyDE expander (lever 3),
and an optional judge (lever 6) — each degrading to lexical/no-op if the aux client is
unavailable, so a missing embedder never breaks live compaction. Defaults: embeddings +
rerank ON; HyDE + semantic-verifier opt-in. Levers 2 & 5 not shipped (cap=4 retained).
77 unit tests green.

## Open (iteration 3 candidate)
**Abstain-on-low-retrieval-confidence** gate: when the top evidence's similarity to the
question is below a threshold (a strong adversarial/unanswerable signal), refuse
deterministically — to push the weak-model residual ~6% toward 0% without a judge call.

## Per-lever detail
`iter2-lever1-embeddings.md`, `iter2-lever3-queryexpansion.md`, `iter2-lever4-reranking.md`,
`iter2-lever5-higherk.md`, `iter2-lever6-semantic-verifier.md`, `iter2-lever2-toolloop.md`.
