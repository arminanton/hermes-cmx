# iter-4 retrieval-lever benchmark — gpt-5-mini, locked config + LLM judge (claude-opus-4.7)
_pooled over seeds [0, 1, 2], n=15 ans + 10 adv each. Goal: ↑acc, halluc not worse._

| arm | ans | acc (judge) | acc (token) | adv_refusal | halluc | Δacc vs base |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 45 | 57.8% | 57.8% | 93.3% | 5.9% | +0.0pt |
| temporal | 45 | 55.6% | 60.0% | 93.3% | 6.1% | -2.2pt |
| importance | 45 | 55.6% | 62.2% | 93.3% | 9.1% | -2.2pt |
| both | 45 | 57.8% | 66.7% | 93.3% | 6.1% | +0.0pt |
