# Iteration 3 — L5 (self-consistency refusal): KEEP (opt-in)

## A/B (gpt-5-mini-low, full ship config, n=18+12, 2 resamples @ temp 0.7)
| self_consistency | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| 0 (off) | 61.1% (11/18) | 83.3% (10/12) | 11.8% (2/17) |
| **2** | **66.7% (12/18)** | **91.7% (11/12)** | **6.2% (1/16)** |

## Verdict: KEEP (opt-in; cost = N× generation on factual turns)
Resampling at temperature>0 and refusing when the salient fact DRIFTS catches confabulations
(an on-topic-but-unanswerable answer is unstable) without over-refusing grounded answers
(anchored by evidence → stable): adversarial refusal 83→92%, hallucination 11.8→6.2%, no
accuracy cost. Same magnitude as L2 — likely partially overlapping; both target the residual.
Cost makes it opt-in (`cfg.self_consistency`), best reserved for the weakest models.
