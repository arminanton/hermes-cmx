# Results map

Every benchmark output, grouped by theme. Headline numbers and verdicts are summarized in
[`../README.md`](../README.md); this is the file-by-file index.

> Reminder: per-conversation variance is large — read these as **pooled** signals, not single
> data points. Numbers were produced under the Hermes `pysqlite3` interpreter.

## Head-to-head vs LCM (the "is it better than LCM?" answer)
| File | What it shows |
|---|---|
| `vs-lcm-gpt-5-mini-low.md` | **cmx vs LCM on identical LOCOMO questions**: cmx 66.7%/100%/**0%** vs LCM 58.3%/83.3%/**10%** (acc / adversarial-refusal / hallucination) |

## Live / real-model proofs
| File | What it shows |
|---|---|
| `real-models.md` | grounding contract across opus-4.7 / sonnet-4.5 / gpt-5.5 / gpt-5-mini (synthetic suite) |
| `h2-live-smoke.md` | the H2 host enforcement hook smoke-tested live on all four providers |

## Iteration 2 — retrieval & safety levers
| File | What it shows |
|---|---|
| `iter2-FINAL-assessment.md` | the 6-lever verdict table + ship config (recall@8 55→82%) |
| `iter2-lever1-embeddings.md` | semantic kNN — SHIP |
| `iter2-lever2-toolloop.md` | model-driven retrieval — REJECT (halluc 27–39%) |
| `iter2-lever3-queryexpansion.md` | HyDE — SHIP opt-in |
| `iter2-lever4-reranking.md` | embedding-cosine rerank — SHIP |
| `iter2-lever5-higherk.md` | higher inject_k — REJECT |
| `iter2-lever6-semantic-verifier.md` | NLI verifier — SHIP opt-in |
| `locomo-iter2-lever1-embeddings.md` | the embeddings lever measured on LOCOMO |

## Iteration 3 — driving the residual hallucination down
| File | What it shows |
|---|---|
| `iter3-FINAL-assessment.md` | the L1–L6 verdict table; L2 sufficiency gate is the keeper ★ |
| `iter3-HARDENING-conclusion.md` | **the honesty file** — at larger n the residual is real (5–31% on adversarial); early 0% was noise |
| `iter3-hardened-gate-RESULT.md` | the hardened L2 gate result |
| `iter3-hardened-stack-REGRESSION.md` | proof that stacking more levers made it *worse* |
| `iter3-L1-abstain-REJECTED.md` | similarity-threshold abstain — REJECT |
| `iter3-L2-sufficiency.md` | the sufficiency pre-gate — KEEP |
| `iter3-L3-chunking.md` | chunk-level embeddings — REJECT |
| `iter3-L4-llm-rerank.md` | LLM reranker — KEEP opt-in |
| `iter3-L5-self-consistency.md` | self-consistency refusal — KEEP opt-in |
| `iter3-L6-graded.md` | graded sufficiency — REJECT (binary wins) |
| `iter3-exp1-foreground-reasoning.md` | effect of foreground reasoning on/off |
| `iter3-exp4-strict-value.md` | strict value-grounding experiment |

## Iteration 4 — multi-judge (Council) & forcing (`matrix/`)
| File | What it shows |
|---|---|
| `matrix/iter4-FINAL-RECOMMENDATION.md` | the final-product config + why Council was rejected |
| `matrix/iter4-council-matrix.md` | Council-as-judge matrix (over-refuses to 10%, 15× cost) |
| `matrix/iter4-FINALISTS.md` | finalist configs head-to-head |
| `matrix/iter4-GATE-SWEEP.md` | sufficiency-gate threshold sweep |
| `matrix/iter4-FORCING.md` | forcing F1–F5 — REJECT (−40pt on cooperative models) |
| `matrix/iter4-DIAGNOSTIC.md` | window-decoupling & single/multi-hop diagnostic |
| `matrix/*.jsonl` | raw per-question records behind the tables |

## Judge-strength ablation
| File | What it shows |
|---|---|
| `judge-ablation-SUMMARY.md` | does a stronger judge reduce hallucination? (summary) |
| `judge-ablation-opus47.md` / `judge-ablation-gpt5.4.md` / `judge-ablation-mini.md` | per-judge runs — strength matters for the *gate*, not for citation *rescue* |

## Per-model matrix runs (pooled LOCOMO)
`matrix-mini-low.md`, `matrix-mini-med.md`, `matrix-opus46fast.md`, `matrix-opus46med.md`,
`matrix-opus47med.md`, `matrix-sonnet46.md`, `matrix-sonnet46med.md`, `matrix-gemini31.md`
— the locked stack measured per foreground model/reasoning setting. Counter-intuitive finding:
**gpt-5-mini-low is among the best** at unanswerability detection; **sonnet-4.6 with reasoning
off is the worst** — raw model strength does not predict grounding.

## Rolling pointers
`latest.md`, `locomo.md` — the most-recent run snapshots used during iteration.
