# Iteration 3 — L6 (graded sufficiency confidence): REJECTED (binary L2 is better)

## A/B (gpt-5-mini-low, gpt-5.4 judge, n=18+12)
| sufficiency mode | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| **binary YES/NO (L2)** | **66.7%** | **100%** | **0%** |
| graded 0-100, refuse <50 (L6) | 61.1% | 91.7% | 6.2% |

## Verdict: REJECT
The graded confidence is worse on all three metrics. A binary YES/NO sufficiency decision is
a cleaner judgment than a poorly-calibrated 0-100 score thresholded at 50 (the score adds
noise without signal). Binary L2 already hits 100%/0% on the weakest model — no headroom for
a graded knob to improve. Keep the binary sufficiency gate.
