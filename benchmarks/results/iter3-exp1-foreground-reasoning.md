# Iteration 3 — Experiment 1 (foreground reasoning ON vs OFF, sequential)

Hardened sufficiency gate fixed (reasoning+vote3 judge), n=15 answerable + 10 adversarial.
Ran sequentially to avoid provider cooldowns.

| model | config | answerable acc | adversarial refusal | hallucination |
|---|---|---:|---:|---:|
| sonnet-4.6 | reasoning OFF | 73.3% | 40.0% | 35.3% |
| sonnet-4.6 | reasoning MED | 66.7% | 50.0% | 31.2% |
| opus-4.6 | reasoning OFF ("fast") | 73.3% | 80.0% | 20.0% |
| opus-4.6 | reasoning MED | 86.7% | 80.0% | 18.8% |

## Read
- Foreground reasoning helps grounding stability for these Claude models (hallucination
  down in both pairs), with a clear accuracy gain on opus-4.6.
- Sonnet-4.6 still has a large residual either way; reasoning improves safety modestly but
  does not solve it.
- Keep reasoning enabled where latency permits; this is a net-positive lever, especially on
  opus-4.6.
