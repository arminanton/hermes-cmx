# iter-4 FINALISTS — gpt-5-mini foreground, n=10 ans + 24 adv, council=claude-opus-4.7
_Decision: 0% halluc guardrail → max adv-refusal → max accuracy → min latency._

| judge config | answer_acc | adv_refusal | halluc | latency |
|---|---:|---:|---:|---:|
| single:opus|gate | 70.0% | 91.7% | 20.0% | 271s |
| single:opus|verify-only | 70.0% | 70.8% | 43.8% | 246s |
| council:m5cc|gate | 10.0% | 95.8% | 50.0% | 4127s |
| council:m5cc|verify-only | 60.0% | 70.8% | 43.8% | 207s |
