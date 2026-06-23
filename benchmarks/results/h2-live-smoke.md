# H2 Option A — live per-provider wiring smoke (2026-06-07)

Exercises the exact code path the conversation_loop wiring runs, against each live model.

| Provider | accept valid answer | replace fabrication | WIRING | model answered correctly |
|---|---|---|---|---|
| opus-4.7    | yes | yes | PASS | yes |
| sonnet-4.5  | yes | yes | PASS | yes |
| gpt-5.5     | yes | yes | PASS | **no — conservatively refused (safe: no hallucination)** |
| gpt-5-mini  | yes | yes | PASS | yes |

ALL PROVIDERS PASS the wiring. Enforcement accepts grounded/refusal answers and
replaces fabrications on every model. Finding: gpt-5.5 (copilot) under-used the
injected evidence and refused an answerable question — a candidate for
provider-specific contract tuning, but the safety property holds (refuse > confabulate).
