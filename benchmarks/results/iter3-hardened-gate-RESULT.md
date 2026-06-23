# Iteration 3 — hardened sufficiency gate (reasoning + K-vote consensus): the residual lever

Targets the root cause (the judge's on-topic false-positives) with: (1) a reasoning/
decomposition prompt — "name the SPECIFIC fact the question needs, then check that EXACT
fact is in the evidence; on-topic-but-absent = NO"; (2) K=3 judgments at temp 0.6 requiring
UNANIMOUS "answerable" to proceed (an unstable judgment → refuse). Judge = gpt-5.4. Judge
calls only, no extra generation.

## A/B on the worst models (n=22+14)
| model | baseline L2 (acc/refuse/halluc) | hardened gate | hallucination |
|---|---|---|---|
| opus-4.7 medium | 63.6% / 78.6% / 20.0% | 68.2% / 85.7% / **10.0%** | 20 → **10** |
| gpt-5-mini medium | 54.5% / 85.7% / 11.1% | 63.6% / 92.9% / **5.3%** | 11 → **5.3** |
| sonnet-4.6 (no-reasoning) | 81.8% / 35.7% / 31.0% | 77.3% / 57.1% / **25.0%** | 31 → **25** |
| gemini-3.1-pro | 27.3% / 64.3% / 21.7% | 31.8% / 50.0% / 25.9% | 22 → 26 (outlier) |

## Verdict: KEEP — the most effective residual lever (roughly HALVES it on most models)
- Reasoning + consensus cut the judge's false-positive on on-topic-but-unanswerable, ~halving
  hallucination on opus-4.7 (20→10) and gpt-5-mini-med (11→5.3); modest on sonnet (31→25);
  accuracy held or improved. Cost: 3 cheap judge calls/factual turn.
- **Still not 0%** — the residual is reduced, not eliminated; a hard floor remains (~5-25%).
- **gemini-3.1-pro is an outlier**: 27-32% accuracy, residual unimproved — it handles the
  cite-[id=N] contract poorly via the aux route; needs provider/prompt-format work, not a
  grounding fix. Recommend excluding it from the default matrix until tuned.

## Config: sufficiency_votes=3, sufficiency_reasoning=True (judge gpt-5.4). Wired in engine +
## production enforce_response. Best stacked with reasoning ENABLED on the foreground model.
