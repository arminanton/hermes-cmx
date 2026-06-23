# Iteration 3 — hardened full stack (L2+L4+L5) + aggressive contract: REGRESSION

## Full stack (L2 sufficiency + L4 LLM-rerank + L5 self-consistency=2), n=22+14, STRONG cite contract
| model | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| opus-4.7 medium | 68.2% | 85.7% | 10.5% |
| opus-4.6 fast | 86.4% | 85.7% | 9.1% |
| gpt-5-mini low | 63.6% | 71.4% | 18.2% |
| gpt-5-mini medium | 77.3% | 64.3% | 20.8% |
| sonnet-4.6 | 86.4% | 42.9% | 29.6% |
| gemini-3.1-pro | 50.0% | 71.4% | 16.7% |

vs L2-only (n=12+8, mild contract): opus/mini-low were 0% halluc / 100% adversarial refusal.

## Finding: stacking + over-instruction BACKFIRED (two confounds, both bad)
1. **Aggressive "cite EVERY fact" contract** pushes models to cite-and-answer even
   unanswerable questions → accuracy up, but adversarial refusal DOWN and hallucination UP.
2. **L4 (LLM reranker) undermines L2 (sufficiency gate):** the reranker surfaces more
   relevant-LOOKING evidence for on-topic adversarial questions, so L2's sufficiency judge
   passes them as "answerable" → the gate stops refusing → confabulation.

This vindicates the one-lever-at-a-time discipline: changing the contract AND stacking L4
together produced a regression that isolated levers did not. **Recommendation: ship L2-only
with the mild contract** (validated 0%/100% on opus×2 + mini-low). L5 helped in isolation
(opt-in for the weakest models); L4 is a RECALL lever that must NOT feed the sufficiency
gate's evidence (or must be disabled when the sufficiency gate is on).
