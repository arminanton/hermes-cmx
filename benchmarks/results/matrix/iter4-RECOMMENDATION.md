# iter-4 — Council judge matrix: safe final-product recommendation
_2 completed cells. Decision: 0% halluc guardrail → max adv-refusal → max accuracy → min latency._


## gpt-5-mini · native  (2 configs)

| judge config | answer_acc | adv_refusal | halluc | latency |
|---|---:|---:|---:|---:|
| single:opus | 50.0% | 100.0% | 0.0% | 166s |
| single:mini | 0.0% | 100.0% | 0.0% | 47s |

**→ Recommended for gpt-5-mini (native): `single:opus`** — acc 50.0%, adv-refusal 100.0%, halluc 0.0%, 166s.

## Cross-model final product (safe across the most models, cheapest)

| judge config | models@0%halluc | total latency |
|---|---:|---:|
| single:mini | 1 | 47s |
| single:opus | 1 | 166s |

**→ Provisional final product: `single:mini`** (0% halluc on 1 model-configs at lowest cost). Confirm at higher n on the finalists.
