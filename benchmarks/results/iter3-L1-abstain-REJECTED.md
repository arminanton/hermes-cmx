# Iteration 3 — L1 (abstain-on-low-retrieval-confidence): REJECTED by calibration

## Hypothesis
Adversarial/unanswerable LOCOMO questions (category 5, answer=None) would retrieve
LOW-similarity evidence, so a threshold τ on retrieval confidence (max cosine(query,
evidence)) could refuse them deterministically, before a weak model confabulates.

## Calibration (deterministic, embeddings only, 40 answerable + 40 adversarial)
| set | mean max-cos | p50 | p75 | p90 |
|---|---|---|---|---|
| answerable  | 0.592 | 0.582 | — | — |
| adversarial | 0.569 | 0.574 | **0.615** | **0.670** |

τ sweep (refuse if max-cos < τ):
| τ | answerable kept | adversarial refused |
|---|---|---|
| 0.40 | 100% | 0% |
| 0.50 | 92.5% | 15% |
| 0.55 | 62.5% | 42.5% |
| 0.60 | 45% | 67.5% |

## Verdict: REJECT
The distributions **overlap** — adversarial questions actually have a HIGHER upper tail
(p75 0.615, p90 0.670) than the answerable median (0.582). There is no τ that keeps
answerable while refusing adversarial: catching 68% of adversarial costs 55% of answerable.

## Why (the load-bearing insight for iteration 3)
LOCOMO adversarial questions are **on-topic but unanswerable** — they ask plausible things
about the same people/events that simply were never stated. So they retrieve topically
similar evidence; a similarity signal cannot tell "answerable" from "on-topic-but-absent."
**Killing the residual hallucination requires a SEMANTIC check of evidence *sufficiency*
("does this evidence actually answer THIS question, or is the honest answer 'I don't
know'?"), not a retrieval-confidence threshold.** → promote L2 to the iteration-3 headline.

Corollary: also explains the judge ablation (stronger judge didn't help) — the residual
isn't in the citation-rescue path; it's the model answering an on-topic-but-unanswerable
question from genuinely-similar (but insufficient) evidence.
