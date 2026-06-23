# Iteration 2 — Lever 2: Re-enable the model's retrieval tool loop

## Result (gpt-5-mini, embeddings on, 16 answerable + 6 adversarial)

| config | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| embeddings, no tools | 12.5% (2/16) | **100%** (6/6) | **0%** (0/3) |
| embeddings + tools   | 12.5% (2/16) | **0%** (0/6) | **27.3%** (6/22) |

## Verdict: REJECT (alone) — must be coupled with Lever 6

- Tools gave **no accuracy gain** (12.5% either way) but **destroyed safety**: adversarial
  refusal 100%→0%, hallucination 0%→27%.
- Cause: `cmx_grep` returns nearest-neighbour turns even for genuinely-unanswerable
  questions; the model then **confabulates an answer from loosely-related results**
  instead of refusing. The deterministic citation check passes (it cites a real turn
  that contains the token) but that turn **does not actually answer the question**.
- **This is the model-agency failure mode cmx exists to prevent** — empirical proof that
  *retrieval without enforcement reintroduces hallucination* (the RLM risk; cmx's whole thesis).
- **Implication:** the tool loop is only safe with a **semantic verifier (Lever 6)** that
  checks "does this evidence actually ANSWER the question?", not just "does the cited
  source contain the token?". Re-test Lever 2 ON TOP of Lever 6.
- Code kept but **disabled by default** (`forced_gate="off"`, text-only) — do NOT ship
  the tool loop without Lever 6.
