# iter-4 lever 2 — MULTI-HOP synthesis (claude-opus-4.8, window 8000, judge claude-opus-4.7)
_pooled seeds [0, 1, 2]. halluc(strict)=non-verbatim value (flags correct derivations); halluc(judge)=shipped answer the judge ruled WRONG (fair to multi-hop derivations)._

| arm | single-hop acc | multi-hop acc | adv refusal | halluc(strict) | halluc(judge) |
|---|---:|---:|---:|---:|---:|
| baseline | 76.5% (26/34) | 30.8% (8/26) | 97.2% | 13.3% | 24.4% |
| multihop | 82.4% (28/34) | 30.8% (8/26) | 97.2% | 28.6% | 26.5% |

**Δ multihop vs baseline: multi-hop acc +0.0pt · adv-refusal +0.0pt · halluc(judge) +2.1pt**

_verdict (by FAIR judge-halluc): **REJECT**._
