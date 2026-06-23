# iter-4 FORCING-LAYER benchmark — n=10 ans + 12 adv, seeds [0, 1]
_baseline (locked, no forcing) vs forcing (F1–F5 + C3 judge guard). Goal: ↑accuracy, hallucination not worse._

| foreground | seed | arm | answer_acc | adv_refusal | halluc | latency |
|---|---|---|---:|---:|---:|---:|
| gemini-2.5-pro | 0 | baseline | 70.0% | 91.7% | 11.1% | 209s |
| gemini-2.5-pro | 0 | forcing | 50.0% | 91.7% | 11.1% | 220s |
| gemini-2.5-pro | 1 | baseline | 60.0% | 100.0% | 0.0% | 321s |
| gemini-2.5-pro | 1 | forcing | 70.0% | 100.0% | 12.5% | 243s |
| gpt-5-mini | 0 | baseline | 60.0% | 91.7% | 11.1% | 204s |
| gpt-5-mini | 0 | forcing | 30.0% | 91.7% | 11.1% | 222s |
| gpt-5-mini | 1 | baseline | 80.0% | 100.0% | 0.0% | 222s |
| gpt-5-mini | 1 | forcing | 30.0% | 91.7% | 11.1% | 235s |

## Δ (forcing − baseline), per foreground (pooled seeds)

| foreground | Δ answer_acc | Δ adv_refusal | Δ halluc |
|---|---:|---:|---:|
| gemini-2.5-pro | -5.0pt | +0.0pt | +6.3pt |
| gpt-5-mini | -40.0pt | -4.1pt | +5.5pt |
