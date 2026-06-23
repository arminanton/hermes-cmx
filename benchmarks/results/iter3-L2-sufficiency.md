# Iteration 3 — L2 (evidence-sufficiency pre-gate): mixed (mini judge)

## A/B (gpt-5-mini-low foreground, full ship config, n=18+12, sufficiency judge = gpt-5-mini)
| sufficiency_gate | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| OFF | 66.7% (12/18) | 83.3% (10/12) | 11.8% (2/17) |
| ON  | 55.6% (10/18) | 91.7% (11/12) | 7.1% (1/14) |

## Read
- Safety up modestly: adversarial refusal 83→92%, hallucination 11.8→7.1% (−1 each, noisy).
- Accuracy DOWN 67→56% (over-refusal: the gate refused ~2 genuinely-answerable questions).
- Did NOT reach the ~0% goal. A weak (mini) sufficiency judge mis-calls the on-topic case
  in BOTH directions: refuses some answerable, passes some unanswerable.

## Next
The sufficiency JUDGMENT is a harder reasoning task than the citation-rescue judgment
(where judge strength was irrelevant). Testing a STRONGER sufficiency judge (gpt-5.4) to see
if it shifts the trade favorably — i.e. catches more adversarial WITHOUT over-refusing
answerable. If it still over-refuses, the honest conclusion is that the weak-model residual
(~6-12%) is a hard floor for on-topic-but-unanswerable questions, and the better lever is
foreground-model choice (strong models already hit 0%) rather than a sufficiency gate.

## UPDATE — stronger sufficiency judge (gpt-5.4): KEEP
| config | answerable acc | adversarial refusal | hallucination |
|---|---|---|---|
| baseline (no L2) | 66.7% | 83.3% | 11.8% |
| L2 + mini judge | 55.6% (over-refuse) | 91.7% | 7.1% |
| **L2 + gpt-5.4 judge** | **66.7%** | **91.7%** | **6.2%** |

**Verdict: KEEP (with a CAPABLE sufficiency judge).** Unlike the citation-rescue judge
(strength irrelevant), the sufficiency judgment is a harder reasoning task — a weak judge
over-refuses (acc 56%), a capable judge (gpt-5.4) improves adversarial refusal (83→92%) and
hallucination (11.8→6.2%) at NO accuracy cost. Does not reach 0% alone; stack with other
levers. Recommend verifier_model = gpt-5.4 (foreground-tier) when sufficiency_gate is on.
