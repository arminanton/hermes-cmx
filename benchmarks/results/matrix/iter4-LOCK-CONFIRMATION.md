# iter-4 LOCK CONFIRMATION — gpt-5-mini foreground, n=20 ans + 30 adv × seeds [0, 1, 2]
_Pooled across seeds (raw-count aggregation). 0% halluc guardrail → max adv-refusal → max acc → min latency._


## gate=opus-4.7

| seed (sample) | answer_acc | adv_refusal | halluc | latency |
|---|---:|---:|---:|---:|
| 0 | 55.0% | 90.0% | 17.6% | 414s |
| 1 | 70.0% | 95.8% | 6.2% | 366s |
| 2 | 50.0% | 100.0% | 0.0% | 416s |

**POOLED (60 ans + 84 adv, 45 shipped): acc 58.3% · adv-refusal 95.2% · halluc 8.9%**

## gate=gpt-5-mini

| seed (sample) | answer_acc | adv_refusal | halluc | latency |
|---|---:|---:|---:|---:|
| 0 | 40.0% | 90.0% | 20.0% | 391s |
| 1 | 50.0% | 87.5% | 18.8% | 349s |
| 2 | 35.0% | 90.0% | 28.6% | 425s |

**POOLED (60 ans + 84 adv, 45 shipped): acc 41.7% · adv-refusal 89.3% · halluc 22.2%**
