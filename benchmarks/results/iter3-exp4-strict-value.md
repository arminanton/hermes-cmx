# Iteration 3 — Experiment 4 (strict value-grounding backstop)

Config: hardened sufficiency gate (reasoning + 3-vote unanimous, judge=gpt-5.4), n=22+14,
run sequentially.

| model | answerable acc | adversarial refusal | hallucination |
|---|---:|---:|---:|
| opus-4.7 medium | 68.2% | 85.7% | 10.0% |
| gpt-5-mini medium | 63.6% | 92.9% | 5.3% |
| sonnet-4.6 | 77.3% | 42.9% | 29.6% |
| gemini-3.1-pro | 27.3% | 71.4% | 17.4% |

## Verdict
Mixed/mostly neutral. It did not materially improve the residual on the main problem models
(opus/mini/sonnet stayed effectively the same as hardened-gate baseline). It helped gemini a
bit, but gemini remains an outlier with low accuracy and high hallucination. Keep as opt-in
hardening, not as the primary residual lever.
