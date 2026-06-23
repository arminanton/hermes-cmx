# iter-4 Council judge benchmark matrix
_Council judge lane: hermes/claude-opus-4.7 (single-model, many-souls) · 5 cells_

| foreground | window | judge | n(ans/adv) | answer_acc | adv_refusal | halluc | sec |
|---|---|---|---|---:|---:|---:|---:|
| gpt-5-mini | native | council:fast | 8/6 | 12.5% | 100.0% | 0.0% | 822 |
| gpt-5-mini | native | council:fast+peer | 8/6 | 12.5% | 100.0% | 0.0% | 826 |
| gpt-5-mini | native | council:standard | 8/6 | 12.5% | 100.0% | 0.0% | 953 |
| gpt-5-mini | native | single:mini | 2/2 | 0.0% | 100.0% | 0.0% | 47 |
| gpt-5-mini | native | single:opus | 8/6 | 50.0% | 100.0% | 0.0% | 166 |
