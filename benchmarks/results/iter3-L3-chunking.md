# Iteration 3 — L3 (chunk-level embeddings for long turns): REJECTED (no measurable gain)

## Test: buried-fact recall — whole-turn emb-kNN vs chunk emb-kNN (FTS isolated out)
Facts buried in very long turns (~12× filler ≈ 8000 chars; fact <2% of the turn), 60 long
filler distractors, real text-embedding-3-small.

| k | whole-turn embedding | chunk embedding |
|---|---|---|
| 3 | 100% (10/10) | 100% (10/10) |
| 5 | 100% (10/10) | 100% (10/10) |
| 8 | 100% (10/10) | 100% (10/10) |

## Verdict: REJECT (keep code behind default-off flag)
The embedder already retrieves buried distinctive facts perfectly even when they are a tiny
fraction of a long turn — chunking adds a chunk table + extra embedding calls for zero recall
gain on these workloads. The residual hallucination is an on-topic-unanswerable (safety)
problem, not a long-turn precision problem, so L3 does not address it. `use_chunks` stays a
default-off option for extreme long-turn corpora; not shipped on.
