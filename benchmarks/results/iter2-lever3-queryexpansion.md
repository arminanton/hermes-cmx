# Iteration 2 — Lever 3: Query expansion (keyword + HyDE)

## LOCOMO evidence-recall@8 (20 answerable questions, embeddings on)

| config | recall@8 |
|---|---|
| baseline (embeddings) | 60% (12/20) |
| + keyword (de-noise)  | 60% (12/20) — neutral alone |
| **+ HyDE**            | **70% (14/20)** (+10pts) |
| **+ keyword + HyDE**  | **75% (15/20)** (+15pts) |

## Verdict: KEEP (HyDE)
- HyDE (search a *hypothetical answer* sentence, semantically closer to the verbatim
  evidence than the question) adds **+10–15pts recall@8**. Keyword de-noise is neutral
  alone but free and additive with HyDE.
- Cost: one cheap model call per query (gpt-5-mini) — second-biggest recall lever after
  embeddings.
- Cumulative recall@8: v1 ~55% → +embeddings 65% → +HyDE 75%.
