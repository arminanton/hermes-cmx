# Iteration 2 — Lever 1: Embeddings (semantic retrieval)

## Primary metric: LOCOMO evidence-recall@k (40 answerable questions, deterministic)

| k | lexical (FTS5+trigram+graph) | fused (+embeddings) | emb-only |
|---|---|---|---|
| 5  | 52% (21/40) | 57% (23/40) | 52% |
| 8  | 55% (22/40) | **65% (26/40)** | 57% |
| 15 | 57% (23/40) | **72% (29/40)** | 72% |

## Full LOCOMO gate (real models, 8 answerable — noisy at this size)
| model | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| opus-4.7 | 25% (2/8) | 100% | 0% |
| gpt-5-mini | 12.5% (1/8) | 100% | 0% |

## Verdict: KEEP
- Embeddings add **+10pts recall@8 (55→65%)** and **+15pts recall@15 (57→72%)** — a clear,
  meaningful improvement on a proper sample.
- **Fusion > either signal alone** at k=8 → keep all four rank-lists in RRF.
- **Higher-k synergy** is strong (72% at k=15) → previews lever 5 (raise inject_k).
- Guardrail intact: **0% hallucination, 100% adversarial refusal** with embeddings on.
- Embedder: copilot `text-embedding-3-small` (1536-d, batched) via the Hermes aux client;
  numpy brute-force cosine kNN; degrades gracefully if unavailable.
- Note: the **strict profile's inject_k is only 4** (weak/small models) — caps recall for
  exactly the models that most need evidence. Revisit in lever 5 (k tuning).
