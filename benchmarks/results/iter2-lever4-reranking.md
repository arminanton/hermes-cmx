# Iteration 2 — Lever 4: Reranking (embedding-cosine)

## What
After hybrid RRF fusion, re-score the candidate pool (`max(k*4, 20)`) by
`cosine(query_embedding, candidate_embedding)` — reusing the per-session stored
vectors, re-embedding only the query once — so semantically-best evidence is
promoted into the injected top-k. Gated by `cfg.rerank` (default off);
`cfg.rerank_weight` (default 1.0) scales the cosine term added to the fused score.

## recall_diag (40 questions, deterministic; embeddings on)

| config | recall@8 | recall@15 |
|---|---|---|
| embeddings (no rerank)        | 65% (26/40) | 72% (29/40) |
| **+ rerank**                  | **77% (31/40)** (+12pt) | **80% (32/40)** (+8pt) |
| emb + HyDE (no rerank)*       | ~75%        | —           |
| **emb + HyDE + rerank (full stack)** | **82% (33/40)** | **85% (34/40)** |

\* HyDE figure from lever-3 measurement; full-stack measured here with HyDE via gpt-5-mini.

## Verdict: KEEP
- +12pt recall@8 on top of embeddings alone; +7pt on top of HyDE. The levers compose.
- Cumulative recall@8: v1 lexical ~55% → +emb 65% → +HyDE 75% → +rerank **82%**.
- Cost: re-embeds the query once per retrieve (vectors are already stored from lever 1);
  bounded pool → cheap. No extra model call beyond the one query embedding.

## Score-scale note (follow-up)
The cosine term (∈[-1,1], weight 1.0) dominates RRF (~0.05) + recency (0.05) + pin (0.10),
so after rerank the ordering is effectively semantic with the fused priors as a tiebreaker.
This is why recall jumps, but it dilutes the pin prior. Pins are a separate **must-keep**
guarantee enforced at the assembly layer (not via this score nudge); if pins must remain
influential within the reranked order, bump `pin_boost`/`rerank_weight` balance. Tracked,
not blocking — recall is the lever-4 objective and pins have their own inclusion path.

## Tests
68 unit tests green (rerank edit regression-free).
