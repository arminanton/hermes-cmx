# Teaching-directive A/B — gpt-5-mini (copilot), pooled samples [0, 1, 2]
_baseline (teach_retrieval off) vs teach (on). A light directive should NOT regress answer_acc; the heavy forcing bundle did (-40pt) on this same model._

| arm | n | answer_acc | adv_refusal | halluc |
|---|---:|---:|---:|---:|
| baseline | 3 | 60.0% | 96.7% | 3.7% |
| teach | 3 | 53.3% | 93.3% | 7.4% |

**Δ (teach − baseline): acc -6.7pt · adv_refusal -3.4pt · halluc +3.7pt**

## per-sample
| arm | sample | acc | adv | halluc | s |
|---|---|---:|---:|---:|---:|
| baseline | 0 | 60.0% | 90.0% | 11.1% | 186s |
| baseline | 1 | 70.0% | 100.0% | 0.0% | 192s |
| baseline | 2 | 50.0% | 100.0% | 0.0% | 176s |
| teach | 0 | 50.0% | 90.0% | 11.1% | 180s |
| teach | 1 | 60.0% | 90.0% | 11.1% | 209s |
| teach | 2 | 50.0% | 100.0% | 0.0% | 179s |
