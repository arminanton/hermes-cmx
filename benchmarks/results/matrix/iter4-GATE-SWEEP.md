# iter-4 GATE SWEEP — single-judge (n=8 ans + 16 adv, maxed retrieval)
_Efficiency + small-window + cross-model. 0% halluc guardrail → max adv-refusal → max acc → min latency._

| foreground | window | gate model | answer_acc | adv_refusal | halluc | latency |
|---|---|---|---:|---:|---:|---:|
| gemini-2.5-pro | native | claude-opus-4.7 | 37.5% | 93.8% | 16.7% | 218s |
| gemini-2.5-pro | native | gpt-5-mini | 62.5% | 87.5% | 22.2% | 245s |
| gemini-2.5-pro | 16000 | claude-opus-4.7 | 62.5% | 87.5% | 25.0% | 206s |
| gemini-2.5-pro | 16000 | gpt-5-mini | 37.5% | 87.5% | 25.0% | 235s |
| gpt-5-mini | native | claude-opus-4.7 | 37.5% | 87.5% | 25.0% | 202s |
| gpt-5-mini | native | gpt-5-mini | 50.0% | 93.8% | 14.3% | 232s |
| gpt-5-mini | 16000 | claude-opus-4.7 | 50.0% | 87.5% | 25.0% | 192s |
| gpt-5-mini | 16000 | gpt-5-mini | 62.5% | 87.5% | 25.0% | 234s |
