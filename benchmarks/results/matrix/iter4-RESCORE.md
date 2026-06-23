# iter-4 DEFINITIVE re-score — gpt-5-mini, locked config, judge=claude-opus-4.7
_n=15 ans + 15 adv per conversation. token-match vs LLM-judge accuracy._

| seed | answerable | acc (token) | acc (LLM-judge) | adv_refusal | halluc | latency |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 15 | 46.7% | 46.7% | 93.3% | 9.1% | 351s |
| 1 | 15 | 80.0% | 80.0% | 86.7% | 13.3% | 360s |
| 2 | 15 | 46.7% | 40.0% | 100.0% | 0.0% | 324s |
| 3 | 15 | 20.0% | 26.7% | 93.3% | 14.3% | 250s |
| 4 | 15 | 40.0% | 33.3% | 100.0% | 0.0% | 452s |

**POOLED (75 ans + 75 adv): token-acc 46.7% · LLM-judge-acc 45.3% · adv-refusal 94.7% · hallucination 7.7%**

_token-vs-judge delta: -1.3pt (if ~0, our scorer is honest; if large, token-match was undercounting)._
