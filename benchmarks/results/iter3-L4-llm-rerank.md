# Iteration 3 — L4 (LLM-relevance reranker): KEEP (opt-in)

## recall_diag (20 LOCOMO answerable, embeddings on)
| config | recall@8 | recall@15 |
|---|---|---|
| cosine-rerank (iter2 lever 4) | 80% (16/20) | 80% (16/20) |
| **+ LLM-rerank (gpt-5-mini, batched)** | **90% (18/20)** | **90% (18/20)** |

## Verdict: KEEP (opt-in, like HyDE)
A batched LLM relevance judge reads each (query, candidate) PAIR and promotes the right
evidence into the top-k better than the bi-encoder cosine (+10pts recall@8 here; +2/20, so
directional given the sample, but consistent with cross-encoder > bi-encoder precision).
Cost: ONE extra model call per retrieval — acceptable at compaction cadence, opt-in via
`cfg.llm_rerank`. Stacks on top of cosine-rerank (applied to the fused pool, degrades to
score order on failure).
